"""
roteiro_engine.py
------------------
Substitui as funções gerar_titulo() / gerar_roteiro() de uma chamada única por uma
cadeia de 5 estágios (tese -> estrutura -> escrita -> crítica adversarial -> título/descrição).

Motivo: uma única chamada "escreva um roteiro sobre X" tende a produzir resumo com
tom dramático por cima — a textura genérica que qualquer LLM produz quando não é
forçado a assumir uma posição específica. Separar em passes obrigatórios resolve isso
porque cada passe sozinho é uma tarefa mais fácil de fazer bem do que pedir tudo junto.

Não depende de nada específico do generate_video.py — recebe a função de chamada ao
Gemini (com retry já embutido) como parâmetro, pra reusar exatamente o _gemini_generate()
que já existe lá, sem duplicar lógica de retry/backoff.

Uso no generate_video.py:

    from roteiro_engine import gerar_pacote_roteiro

    pacote = gerar_pacote_roteiro(
        tema=tema,
        contexto_nicho=CONTEXTO_NICHO,
        idioma_conteudo=IDIOMA_CONTEUDO,
        instrucao_extra=INSTRUCAO_EXTRA_ROTEIRO,
        documento_estilo=config.get('documento_estilo', []),
        tipo_video=VIDEO_TYPE,
        gemini_generate_fn=_gemini_generate,
    )

    roteiro = pacote['roteiro_texto']       # string corrida, pronta pra TTS (compatível com o resto do código)
    titulo_video = pacote['titulo']         # string, já escolhido entre as variantes
    descricao_extra = pacote['descricao']   # dict {abertura_seo, corpo}
    blocos_roteiro = pacote['roteiro_blocos']  # lista de {bloco, texto} — pra Fase 2 (casar B-roll por bloco)

Se qualquer estágio falhar, cai para geração simples (comportamento antigo) em vez de
quebrar o pipeline — mesma filosofia de fallback que já existe no resto do código.
"""

import json
import re
import traceback

from numeros_extenso import normalizar_texto_para_tts, contem_digitos

# ── Configuração por canal (preenchida por generate_video.py a partir do config.json) ──
# Sem chamar configurar(), tudo abaixo fica inerte e o comportamento é o de sempre.
_CFG = {
    'idioma_numeros': None,   # 'no' liga a normalização determinística de números/abreviações
    'regra_numeros': '',      # texto da regra injetada nos prompts de escrita/reescrita
    'nota_publico': '',       # nota sobre público/país injetada nos prompts de estrutura/escrita
    'chars_min': None,        # alvo de tamanho do roteiro INTEIRO, em caracteres (com espaços)
    'chars_max': None,
    'chars_por_palavra': 6.2, # média (com espaço) usada só pra converter alvo de chars em palavras
}


def configurar(**kwargs):
    """Chamado uma vez por generate_video.py. Chaves desconhecidas são ignoradas."""
    for k, v in kwargs.items():
        if k in _CFG and v is not None:
            _CFG[k] = v


def _regra_numeros():
    return f"\n- {_CFG['regra_numeros']}" if _CFG['regra_numeros'] else ""


def _nota_publico():
    return f"\n{_CFG['nota_publico']}\n" if _CFG['nota_publico'] else ""


def _pos_processar_texto(texto):
    """Rede de segurança: tira dígitos/abreviações/pontuação solta que escaparam do prompt.
    Só age se configurar(idioma_numeros=...) foi chamado."""
    if _CFG['idioma_numeros']:
        return normalizar_texto_para_tts(texto, _CFG['idioma_numeros'])
    return texto

# Segunda camada de defesa contra o roteiro "vazar" direção de produção pro que vira
# fala (ex: um documento_estilo em formato de roteiro profissional, com timecode e
# indicação de trilha/efeito sonoro, sendo copiado literalmente pelo Gemini em vez de
# só inspirar o tom). Mesmo com a instrução reforçada no prompt de gerar_prosa, isso
# roda em CIMA do texto já gerado, pra nunca depender só do modelo "obedecer bem".
_PADROES_DIRECAO_PRODUCAO = [
    re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*[-–—]?\s*', re.IGNORECASE),  # "00:00 - " no início da frase
    re.compile(r'\bzero\s+hora[s]?\b', re.IGNORECASE),                     # "zero hora(s)" (timecode por extenso)
    re.compile(r'^\s*(LOCUTOR|NARRADOR|VOZ|APRESENTADOR)\s*:\s*', re.IGNORECASE),  # rótulo de quem fala
    re.compile(r'\[[^\]]*\]'),                                             # qualquer coisa entre colchetes
    re.compile(
        r'\((?:[^()]*\b(?:sfx|trilha|efeito sonoro|som de|transição|corte para|fade)\b[^()]*)\)',
        re.IGNORECASE
    ),  # parênteses com palavra-chave de direção de produção dentro
]

# Contaminação em PROSA LIVRE (sem colchete/rótulo/parêntese) não dá pra pegar por
# trecho isolado — é a FRASE INTEIRA que é direção de produção (ex: "trilha sonora de
# abertura, um pulso grave, eletrônico, minimalista..."), não uma palavra solta no meio
# de uma frase legítima. Por isso aqui o filtro descarta a SENTENÇA INTEIRA quando ela
# acumula 2+ termos dessa lista — uma sentença real sobre o TEMA do vídeo dificilmente
# bate em 2 desses termos ao mesmo tempo, então o risco de falso positivo é baixo.
_TERMOS_PRODUCAO = [
    'trilha sonora', 'efeito sonoro', 'som ambiente', 'som de clique', 'som de mouse',
    'locutor', 'narrador', 'apresentador', 'sfx', 'fade in', 'fade out', 'corte para',
    'transição de', 'transição x', 'pulso grave', 'nota de piano', 'notas de piano',
    'zero hora', 'tom direto, objetivo', 'sem firulas emocionais',
]


def _sentenca_e_direcao_producao(sentenca):
    sentenca_lower = sentenca.lower()
    acertos = sum(1 for termo in _TERMOS_PRODUCAO if termo in sentenca_lower)
    return acertos >= 2


def sanitizar_direcoes_de_producao(texto):
    """
    Remove qualquer resquício de direção de produção que porventura tenha passado pelo
    prompt (ver aviso em gerar_prosa) — timecode/trilha/efeito sonoro em trecho isolado
    (colchete, parêntese, rótulo) E sentenças inteiras em prosa livre que claramente
    descrevem produção de áudio/vídeo em vez de conteúdo do tema. Aplicado em TODO
    texto de bloco antes de virar narração, em qualquer modo.
    """
    for padrao in _PADROES_DIRECAO_PRODUCAO:
        texto = padrao.sub('', texto)

    sentencas = re.split(r'(?<=[.!?])\s+', texto)
    sentencas_limpas = [s for s in sentencas if not _sentenca_e_direcao_producao(s)]
    texto = " ".join(sentencas_limpas)

    return re.sub(r'\s{2,}', ' ', texto).strip()


def _com_retry_estagio(nome_estagio, func, *args, tentativas=2, **kwargs):
    """
    Roda UM estágio da cadeia (ex: gerar_estrutura_capitulos) com uma segunda chance
    antes de deixar a exceção propagar pra fora.

    BUGFIX (fallback pra 'fallback_simples' acontecendo com frequência e pulando os
    capítulos/curadoria via Telegram inteiros): antes, qualquer erro em QUALQUER
    estágio — inclusive um JSON malformado pontual do Gemini, que é um problema
    conhecidamente intermitente de LLM, não um erro sistemático — derrubava a cadeia
    inteira pro modo de emergência de uma vez, sem chance de recuperação. A esmagadora
    maioria desses erros não se repete numa segunda chamada (o modelo, pedido de novo,
    normalmente devolve um JSON válido). Isso tenta de novo O MESMO estágio até
    `tentativas` vezes antes de desistir e deixar a exceção subir pro try/except maior
    (que aí sim cai pro roteiro de emergência sem capítulos).
    """
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ultimo_erro = e
            if tentativa < tentativas:
                print(f"  ⚠️ {nome_estagio} falhou na tentativa {tentativa}/{tentativas} "
                      f"({e}) — tentando de novo...")
    raise ultimo_erro


def _extrair_json(texto):
    """Mesmo padrão já usado em gerar_titulo() no generate_video.py: acha o primeiro { ... }."""
    texto = texto.strip().replace('```json', '').replace('```', '').strip()
    inicio = texto.find('{')
    fim = texto.rfind('}') + 1
    if inicio == -1 or fim == 0:
        raise ValueError(f"Nenhum JSON encontrado na resposta: {texto[:200]}")
    return json.loads(texto[inicio:fim])


# ============================================================
# ESTÁGIO 1 — TESE
# ============================================================

def gerar_tese(tema, contexto_nicho, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Você é um pesquisador cético, não um narrador. Sua única tarefa é produzir
uma AFIRMAÇÃO específica e refutável sobre o tema abaixo — nunca um resumo ou generalidade.

NICHO: {contexto_nicho}
TEMA: "{tema}"

Regras rígidas:
- A afirmação precisa poder estar ERRADA. Se ninguém discordaria dela, rejeite-a.
- Proibido gerar afirmação do tipo "{tema} é importante" ou "{tema} muda tudo".
- A afirmação deve conectar uma causa específica a um efeito específico, ou revelar uma
  crença comum sobre o tema que é enganosa/incompleta.
- Escreva em {idioma_conteudo}.

Retorne APENAS JSON, neste formato exato:
{{
  "tese": "uma frase, a afirmação específica",
  "por_que_e_contestavel": "o que alguém poderia usar para discordar",
  "reprovar": false
}}

Se genuinamente não for possível gerar uma tese específica para esse tema (é raro),
retorne "reprovar": true e explique o motivo em "tese"."""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


# ============================================================
# ESTÁGIO 2 — ESTRUTURA ARGUMENTATIVA
# ============================================================

def gerar_estrutura(tese_dict, contexto_nicho, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Você recebeu uma tese. Construa o esqueleto argumentativo do vídeo.
NÃO escreva prosa ainda — apenas a estrutura lógica, em {idioma_conteudo}.

NICHO: {contexto_nicho}
TESE: {tese_dict['tese']}
POR QUE É CONTESTÁVEL: {tese_dict['por_que_e_contestavel']}

Retorne APENAS JSON com estas 6 chaves (cada uma uma string curta, 1-2 frases):
{{
  "gancho": "a tensão ou pergunta que abre o vídeo — NÃO pode ser a tese repetida",
  "evidencia_a_favor": "o ponto principal que sustenta a tese",
  "objecao": "o argumento mais forte CONTRA a tese, ou a crença comum que ela contraria",
  "resposta_a_objecao": "como a tese sobrevive à objeção",
  "implicacao": "por que isso importa pra quem está assistindo, hoje, na prática",
  "fechamento": "a ideia final — não uma frase motivacional vaga, algo específico e acionável"
}}"""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


# ============================================================
# ESTÁGIO 3 — ESCRITA (só aqui vira prosa)
# ============================================================

def gerar_prosa(estrutura, contexto_nicho, idioma_conteudo, instrucao_extra,
                 documento_estilo, palavras_alvo, gemini_generate_fn):
    """
    Genérica em relação à estrutura: funciona tanto com a estrutura argumentativa de 6
    chaves (Estágio 2 / modo cadeia_completa) quanto com qualquer outra estrutura em
    blocos, como a devocional de 4 chaves (gerar_estrutura_devocional / modo simples).
    O único contrato é: estrutura é um dict ordenado {chave: descrição}, e cada chave
    vira um bloco do roteiro final, na mesma ordem.
    """
    bloco_estilo = ""
    if documento_estilo:
        exemplos = "\n\n".join(f"- {ex}" for ex in documento_estilo)
        bloco_estilo = f"""
DOCUMENTO DE ESTILO — use isso como referência de TOM E VOZ (vocabulário, cadência,
tipo de imagem usada), NÃO como referência de TAMANHO. Os exemplos abaixo podem ser
mais curtos que a meta de palavras pedida mais adiante — nesse caso, desenvolva mais
os mesmos pontos na mesma voz, em vez de encurtar pra bater com o tamanho do exemplo.

ATENÇÃO: se o documento de estilo tiver vindo de um roteiro de produção profissional,
ele pode conter timecode (ex: "00:00", "01:23"), indicação de trilha sonora/efeito
sonoro (ex: "pulso grave eletrônico", "SFX: clique"), rótulo de quem fala (ex:
"LOCUTOR:", "NARRADOR:") ou direção de cena/câmera. NADA disso é texto pra narrar —
é metadado de produção, não fala. Extraia SÓ as palavras que uma pessoa diria em voz
alta, e IGNORE completamente qualquer marcação técnica, mesmo que isso signifique
reescrever o trecho do zero na mesma voz/tom:
{exemplos}
"""

    linha_extra = f"\n- {instrucao_extra}" if instrucao_extra else ""
    n_blocos = max(1, len(estrutura))
    palavras_por_bloco = max(15, round(palavras_alvo / n_blocos))
    itens_estrutura = "\n".join(f"{i+1}. {chave}: {valor}" for i, (chave, valor) in enumerate(estrutura.items()))
    blocos_exemplo = ",\n    ".join(f'{{"bloco": "{chave}", "texto": "..."}}' for chave in estrutura.keys())

    prompt = f"""Escreva o roteiro de narração seguindo ESTRITAMENTE a estrutura abaixo, em {idioma_conteudo}.

NICHO: {contexto_nicho}
{bloco_estilo}
ESTRUTURA (siga esta ordem — cada item vira um bloco de narração de ~{palavras_por_bloco}
palavras, NÃO um parágrafo curto de 1-2 frases):
{itens_estrutura}

REGRAS OBRIGATÓRIAS:
- Duração alvo: ~{palavras_alvo} palavras no total, distribuídas quase igualmente entre
  os {n_blocos} blocos acima (~{palavras_por_bloco} palavras cada) — isso é mais importante
  que soar "conciso"; desenvolva cada ideia com exemplos e detalhes concretos até chegar lá
- PROIBIDO usar: "mas o que isso realmente significa?", "e é aí que tudo muda",
  "prepare-se para descobrir", ou qualquer frase que serviria em vídeo sobre qualquer
  outro tema do mesmo nicho
- Tom investigativo, não expositivo: pelo menos 2-3 blocos (não só a introdução) devem
  conter uma pergunta provocativa/reflexiva genuína — que reformula o problema, expõe
  uma contradição ou questiona quem se beneficia (ex: "Tem alguma coisa errada nesse
  sistema?", "Quem lucra quando isso acontece?"), nunca uma pergunta retórica vazia
  tipo "você já parou pra pensar?". A pergunta precisa ser específica ao FATO que
  acabou de ser narrado naquele bloco, não genérica ao tema inteiro
- Frases curtas, sem formatação, sem asteriscos, sem emojis
- NÃO mencione apresentador, câmera ou elementos visuais
- PROIBIDO incluir QUALQUER marcação técnica de produção: timecode (nem por extenso,
  tipo "zero hora"), indicação de trilha sonora, indicação de efeito sonoro (SFX),
  nome de transição, rótulo de locutor/narrador, direção de cena. O texto de cada
  bloco tem que ser 100% falável em voz alta do primeiro ao último caractere — se ao
  reler um trecho ele soa como uma instrução PRA alguém produzir o vídeo, em vez de
  uma frase QUE o narrador diria, ele não pode entrar no roteiro
- PROIBIDO usar siglas, abreviações ou qualquer atalho de letras pra se referir a um
  país, órgão, lei ou instituição — o texto vai direto pra um TTS que lê tudo de forma
  LITERAL, letra por letra, então "EUA" sai como "É-Ú-A" em vez de "Estados Unidos".
  Escreva sempre por extenso ("Estados Unidos", "Organização das Nações Unidas",
  "Produto Interno Bruto"), mesmo que isso repita a expressão várias vezes ao longo do
  bloco. NUNCA invente uma abreviação curta (tipo uma letra ou duas) pra evitar repetir
  um termo — se precisar variar, troque por uma expressão equivalente por extenso
  ("o país", "a potência norte-americana", "o mercado americano"), nunca por uma sigla
  ou código{_regra_numeros()}{linha_extra}
{_nota_publico()}
Retorne APENAS JSON:
{{
  "blocos": [
    {blocos_exemplo}
  ]
}}"""

    resposta = gemini_generate_fn(prompt)
    dados = _extrair_json(resposta.text)
    blocos = dados['blocos']

    padrao_prefixo = re.compile(r'^\s*\[\d+\]\s*\([^)]*\)\s*:\s*')
    for b in blocos:
        b['texto'] = re.sub(r'\*+', '', b['texto'])
        b['texto'] = b['texto'].replace('#', '').replace('_', '').strip()
        b['texto'] = padrao_prefixo.sub('', b['texto']).strip()
        b['texto'] = _pos_processar_texto(sanitizar_direcoes_de_producao(b['texto']))

    total_palavras = sum(len(b['texto'].split()) for b in blocos)
    if total_palavras < palavras_alvo * 0.85:
        print(f"  ⚠️ Roteiro saiu com {total_palavras} palavras (meta: ~{palavras_alvo}) — "
              f"tentando expandir os blocos mais curtos antes de seguir...")
        blocos = _expandir_blocos_curtos(blocos, palavras_por_bloco, palavras_alvo,
                                          idioma_conteudo, gemini_generate_fn)
        total_palavras = sum(len(b['texto'].split()) for b in blocos)
        if total_palavras < palavras_alvo * 0.7:
            print(f"  ⚠️ Mesmo após expansão, roteiro ficou com {total_palavras} palavras "
                  f"(meta: ~{palavras_alvo}) — o vídeo final vai ficar mais curto que o "
                  f"esperado. Se isso persistir, considere reduzir os exemplos de "
                  f"'documento_estilo' ou torná-los mais longos.")
        else:
            print(f"  ✅ Roteiro expandido para {total_palavras} palavras.")

    return blocos


def _expandir_blocos_curtos(blocos, palavras_por_bloco, palavras_alvo, idioma_conteudo,
                             gemini_generate_fn, tentativas=2):
    """
    LLMs pedidos por "~N palavras" numa única chamada, em especial gerando JSON
    estruturado, tendem a SUBESTIMAR o tamanho pedido — é um padrão conhecido, não um
    erro pontual. Em vez de só avisar no log e entregar um roteiro raso (o que estava
    acontecendo antes), aqui os blocos que saíram abaixo da meta voltam pro modelo numa
    chamada de reescrita cirúrgica (mesmo padrão de indexação de criticar_e_reescrever),
    pedindo especificamente mais profundidade/detalhe/fatos concretos — nunca "encher
    linguiça" com frase genérica — até chegar perto da meta de palavras por bloco.
    Roda no máximo `tentativas` rodadas pra não entrar em loop se o modelo insistir em
    devolver blocos curtos.
    """
    limite_bloco = palavras_por_bloco * 0.85

    for _ in range(tentativas):
        curtos = [
            (i, b) for i, b in enumerate(blocos)
            if len(b['texto'].split()) < limite_bloco
        ]
        if not curtos:
            break

        trechos = "\n".join(
            f"[{i}] ({b['bloco']}, {len(b['texto'].split())} palavras, meta "
            f"~{palavras_por_bloco}): {b['texto']}"
            for i, b in curtos
        )

        prompt = f"""Os blocos de roteiro abaixo saíram MAIS CURTOS do que a meta de
palavras. Reescreva cada um, mantendo o mesmo sentido e a mesma voz, mas DESENVOLVENDO
mais a ideia até chegar perto da meta de palavras indicada — acrescente exemplos,
dados concretos (número, data, nome de lugar/pessoa/instituição/lei), desdobramentos
da mesma ideia ou uma segunda camada de explicação. NUNCA "encha linguiça" com frase
genérica só pra bater a contagem — se não houver mais nada específico a acrescentar
sobre aquele ponto, desenvolva uma implicação prática ou uma comparação concreta dele.
Escreva em {idioma_conteudo}.

{trechos}

REGRAS: nunca use siglas/abreviações pra país/órgão/instituição (escreva por extenso,
ex: "Estados Unidos", nunca "EUA" ou qualquer código curto) — o texto vai pra um TTS
que lê tudo literalmente.{_regra_numeros()}
"texto_novo" deve ser só o texto puro que vai ser narrado,
sem o prefixo "[n] (bloco):" usado acima, sem colchetes, sem marcação técnica de
produção (timecode, trilha, SFX, rótulo de locutor).

Retorne APENAS JSON: {{"expansoes": [{{"indice": 0, "texto_novo": "..."}}]}}"""

        try:
            resposta = gemini_generate_fn(prompt)
            expansoes = _extrair_json(resposta.text).get('expansoes', [])
        except Exception as e:
            print(f"  ⚠️ Expansão de bloco curto falhou ({e}) — mantendo texto atual")
            break

        padrao_prefixo = re.compile(r'^\s*\[\d+\]\s*\([^)]*\)\s*:\s*')
        houve_ganho = False
        for exp in expansoes:
            i = exp.get('indice')
            if not (isinstance(i, int) and 0 <= i < len(blocos) and exp.get('texto_novo')):
                continue
            texto_novo = re.sub(r'\*+', '', exp['texto_novo'])
            texto_novo = texto_novo.replace('#', '').replace('_', '').strip()
            texto_novo = padrao_prefixo.sub('', texto_novo).strip()
            texto_novo = _pos_processar_texto(sanitizar_direcoes_de_producao(texto_novo))
            # Só aceita se de fato ficou mais longo que o texto atual — nunca troca por
            # algo mais curto ou igual (isso indicaria que o modelo não expandiu nada).
            if len(texto_novo.split()) > len(blocos[i]['texto'].split()):
                blocos[i]['texto'] = texto_novo
                houve_ganho = True

        if not houve_ganho:
            break

    return blocos


# ============================================================
# AJUSTE DE TAMANHO POR CARACTERES (alvo do roteiro INTEIRO, ex: 6000–7000)
# ============================================================

def _total_chars(blocos):
    return len(" ".join(b['texto'] for b in blocos))


def _cortar_frases_excedentes(blocos, chars_max):
    """Último recurso quando o Gemini não conseguiu encurtar: remove a ÚLTIMA frase do
    maior bloco do MEIO (nunca a introdução nem o desfecho, que carregam abertura e
    pergunta final) até caber. Só roda se, mesmo após as reescritas, ainda estourar."""
    candidatos = list(range(1, len(blocos) - 1)) if len(blocos) > 2 else list(range(len(blocos)))
    cortes = 0
    while _total_chars(blocos) > chars_max:
        elegiveis = [i for i in candidatos
                     if len(re.split(r'(?<=[.!?])\s+', blocos[i]['texto'].strip())) > 3]
        if not elegiveis:
            break
        i = max(elegiveis, key=lambda k: len(blocos[k]['texto']))
        frases = re.split(r'(?<=[.!?])\s+', blocos[i]['texto'].strip())
        blocos[i]['texto'] = " ".join(frases[:-1])
        cortes += 1
    if cortes:
        print(f"  ✂️ Corte de emergência: {cortes} frase(s) removida(s) dos blocos do meio pra caber no limite")
    return blocos


def ajustar_tamanho_por_caracteres(blocos, idioma_conteudo, gemini_generate_fn, tentativas=3):
    """
    O prompt pede um nº de PALAVRAS, mas o alvo real é em CARACTERES (config
    'caracteres_alvo_min/max'). Palavras em norueguês variam muito de tamanho (compostos),
    então a conversão palavras→caracteres erra com frequência — por isso aqui o tamanho é
    MEDIDO no texto final (já com números por extenso) e, se estiver fora da faixa, os
    blocos voltam pro Gemini com uma meta de caracteres por bloco, proporcional.
    """
    cmin, cmax = _CFG['chars_min'], _CFG['chars_max']
    if not cmin or not cmax:
        return blocos
    alvo = (cmin + cmax) // 2
    padrao_prefixo = re.compile(r'^\s*\[\d+\]\s*\([^)]*\)\s*:\s*')

    for rodada in range(1, tentativas + 1):
        total = _total_chars(blocos)
        if cmin <= total <= cmax:
            print(f"  📏 Roteiro com {total} caracteres — dentro da faixa {cmin}–{cmax} ✅")
            return blocos

        fator = alvo / max(total, 1)
        acao = "ENCURTE" if total > cmax else "DESENVOLVA MAIS"
        print(f"  📏 Roteiro com {total} caracteres (faixa {cmin}–{cmax}) — rodada {rodada}/{tentativas}: {acao.lower()}...")
        metas = {i: round(len(b['texto']) * fator) for i, b in enumerate(blocos)}
        trechos = "\n".join(
            f"[{i}] ({b['bloco']}, {len(b['texto'])} caracteres, META ~{metas[i]} caracteres): {b['texto']}"
            for i, b in enumerate(blocos)
        )
        prompt = f"""{acao} cada bloco do roteiro abaixo até chegar perto da META de caracteres
indicada (contando espaços). Mantenha o mesmo sentido, a mesma voz e TODOS os fatos
importantes; ao encurtar, corte repetição e floreio, nunca o dado concreto nem a pergunta
final; ao desenvolver, acrescente fato/exemplo concreto, nunca frase genérica. Escreva em
{idioma_conteudo}. A meta é aproximada: erre por menos de 5%.

{trechos}

REGRAS: texto 100% falável (sem marcação de produção, sem colchetes); nenhuma sigla; sem
travessão solto.{_regra_numeros()}
"texto_novo" é só o texto puro do bloco, sem o prefixo "[n] (bloco):".

Retorne APENAS JSON com TODOS os blocos: {{"reescritas": [{{"indice": 0, "texto_novo": "..."}}]}}"""

        try:
            reescritas = _extrair_json(gemini_generate_fn(prompt).text).get('reescritas', [])
        except Exception as e:
            print(f"  ⚠️ Ajuste de tamanho falhou ({e}) — mantendo texto atual")
            break

        for r in reescritas:
            i = r.get('indice')
            if not (isinstance(i, int) and 0 <= i < len(blocos) and r.get('texto_novo')):
                continue
            novo = padrao_prefixo.sub('', re.sub(r'\*+', '', r['texto_novo']).strip())
            novo = _pos_processar_texto(sanitizar_direcoes_de_producao(novo.replace('#', '').strip()))
            # só aceita se chegou MAIS PERTO da meta do bloco que o texto anterior
            if novo and abs(len(novo) - metas[i]) < abs(len(blocos[i]['texto']) - metas[i]):
                blocos[i]['texto'] = novo

    total = _total_chars(blocos)
    if total > cmax:
        blocos = _cortar_frases_excedentes(blocos, cmax)
        total = _total_chars(blocos)
    if total < cmin:
        print(f"  ⚠️ Roteiro ficou com {total} caracteres, abaixo do mínimo de {cmin} — "
              f"o vídeo sairá um pouco mais curto que o pedido.")
    else:
        print(f"  📏 Roteiro final com {total} caracteres.")
    return blocos


# ============================================================
# ESTÁGIOS 1-2 (MODO SIMPLES) — sem forçar tese contestável/objeção
# ============================================================

def gerar_estrutura_devocional(tema, contexto_nicho, idioma_conteudo, gemini_generate_fn):
    """
    Alternativa aos Estágios 1+2 (tese + estrutura argumentativa) pra nichos onde
    forçar uma afirmação contestável soa artificial — ex: reflexão devocional, onde
    o valor não vem de defender uma tese contra uma objeção, vem de uma virada de
    perspectiva sobre algo familiar. Ainda produz uma estrutura EM BLOCOS (isso é o
    que a Fase 2 de produção visual precisa pra casar B-roll/destaque por trecho) —
    só que sem contradição forçada.
    """
    prompt = f"""Construa o esqueleto de uma reflexão curta sobre o tema abaixo, em {idioma_conteudo}.
NÃO escreva prosa ainda — apenas a estrutura.

NICHO: {contexto_nicho}
TEMA: "{tema}"

Regras:
- Evite abrir com pergunta genérica tipo "você já parou pra pensar..."
- A reflexão central deve ter um ângulo específico sobre o tema, não uma generalidade
  que serviria pra qualquer tema parecido
- A aplicação prática deve ser concreta (algo pra fazer/observar hoje), não vaga

Retorne APENAS JSON com estas 4 chaves:
{{
  "abertura": "uma imagem, cena cotidiana ou observação concreta que introduz o tema",
  "reflexao_central": "a ideia principal sobre o tema, com um ângulo específico",
  "aplicacao_pratica": "algo concreto que a pessoa pode fazer ou observar hoje",
  "fechamento": "uma frase final que não seja um clichê motivacional vago"
}}"""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


# ============================================================
# ESTÁGIOS 1-2 (MODO WEBDOC EM CAPÍTULOS) — formato investigativo tipo
# "Elementar": introdução com dado forte, N capítulos nomeados (cada um cobrindo
# um ÂNGULO diferente do tema, não um passo de argumentação), desfecho.
# ============================================================

def gerar_estrutura_capitulos(tema, contexto_nicho, idioma_conteudo, gemini_generate_fn, num_capitulos=3):
    """
    Estrutura alternativa à argumentativa (tese/objeção) e à devocional (abertura/
    reflexão) — aqui a lógica é jornalística/investigativa: cada capítulo cobre uma
    FACETA distinta do tema (origem do problema, como funciona hoje, quem é afetado,
    comparação com outro caso etc.), não um degrau de um argumento.

    O título de cada capítulo (curto, tipo rótulo de seção de documentário — "A
    Fábrica de Prédios", "O Esquema do Repasse") é o que vira o CARD PRETO de
    transição entre capítulos no vídeo final, e também dá nome à pasta/critério de
    música tema daquele trecho (ver generate_video.py → gerar_card_capitulo /
    _mixar_musica_por_capitulo).
    """
    prompt = f"""Você é roteirista de documentário investigativo de dados (estilo canais
como "Elementar" no YouTube). NÃO escreva prosa ainda — apenas o esqueleto, em {idioma_conteudo}.

NICHO: {contexto_nicho}
TEMA: "{tema}"
{_nota_publico()}
A diferença entre um webdoc investigativo de verdade e um resumo genérico é esta: o
webdoc ANCORA a explicação abstrata em UM CASO CONCRETO — data, número, nome de lugar
ou instituição — e usa esse caso como fio condutor que atravessa o vídeo inteiro, não
só a introdução. Exemplo do padrão esperado (tema: segurança contra incêndio em
prédios): o vídeo NÃO abre dizendo "normas de segurança são importantes" — abre
contando o incêndio do Edifício Joelma, São Paulo, 1º de fevereiro de 1974, 8h50, 187
mortos de 756 pessoas — e cada capítulo depois volta a ESSE caso pra ilustrar um ponto
diferente (por que não havia sprinkler, o que mudou depois, o que outro país fez
diferente no mesmo período).

Antes de montar os capítulos, IDENTIFIQUE um caso/evento/incidente específico e real
(ou um dado concreto — uma cidade, empresa, lei com número, ano) que sirva de
ANCORAGEM pro tema. Esse caso:
- precisa ser plausível e checável — se você não tiver certeza absoluta de um número
  exato, escreva o dado como o tipo de fato que existe e marque "[confirmar número
  exato]" no lugar dele, em vez de inventar uma estatística falsa apresentada como certa
- precisa reaparecer em pelo menos 2 dos {num_capitulos} capítulos, não só na introdução

Construa {num_capitulos} capítulos, cada um cobrindo uma FACETA DIFERENTE do tema,
todos conectados ao caso de ancoragem.

Regras:
- O título de cada capítulo tem 3 a 6 palavras, como um RÓTULO de seção de documentário
  — não é uma frase completa (ex: "A Fábrica de Prédios", não "Por que os prédios são
  construídos assim")
- A introdução abre CONTANDO o caso de ancoragem como cena (data, hora, número), não
  com uma afirmação genérica sobre o tema, e TERMINA com 1-2 perguntas provocativas/
  reflexivas que reformulam o tema — o padrão é "e se o problema real não for X, mas
  Y?" ou "quem ganha quando Z acontece?", nunca uma pergunta óbvia que qualquer um já
  responderia. É essa pergunta que vira o fio condutor que o resto do vídeo responde.
- Em "cobre", diga EXPLICITAMENTE que fato/número/nome específico aquele capítulo vai
  usar como evidência — não só o ângulo abstrato (isso é instrução pra quem for
  escrever a prosa depois não fugir pro genérico) — e, quando fizer sentido, que
  pergunta provocativa esse capítulo levanta ou responde
- O desfecho responde à pergunta implícita do vídeo citando de novo o caso de ancoragem,
  e termina devolvendo a pergunta pro espectador de um jeito específico (não "e você, o
  que acha?" genérico — algo que só faz sentido pra ESTE tema, ex: "e você, acha que
  falta água mesmo, ou falta vontade de resolver o problema de verdade?")

Retorne APENAS JSON:
{{
  "caso_ancoragem": "descrição do caso/evento específico escolhido, com os dados que você tem certeza + [confirmar] onde não tiver",
  "introducao": "o que a introdução deve cobrir — o caso de ancoragem contado como cena + apresentação do tema",
  "capitulos": [
    {{"titulo": "Título Curto do Capítulo 1", "cobre": "que faceta/ângulo este capítulo aborda + que fato/número específico usa como evidência"}}
  ],
  "desfecho": "o que o desfecho deve responder/concluir, citando de novo o caso de ancoragem"
}}
(o array "capitulos" deve ter exatamente {num_capitulos} itens)"""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


def gerar_estrutura_lista(tema, contexto_nicho, idioma_conteudo, gemini_generate_fn,
                          num_itens=7, itens_fornecidos=None, ordem='regressiva'):
    """
    Estrutura de vídeo em LISTA ("10 motos que...", "7 comidas que..."): introdução → N itens
    → desfecho. Devolve o MESMO formato de gerar_estrutura_capitulos (chave 'capitulos' =
    os itens; 'caso_ancoragem' = o critério da lista), então toda a escrita/crítica/ajuste
    de tamanho/montagem reaproveita o pipeline dos capítulos sem mudança.

    itens_fornecidos: nomes de itens escolhidos pelo usuário (na ORDEM em que aparecem no
    vídeo). Se vier, o Gemini NÃO inventa/renomeia — só escreve o que cada item cobre.
    ordem: 'regressiva' (N→1, o último item mostrado é o nº 1) ou 'crescente' (1→N).
    """
    n = len(itens_fornecidos) if itens_fornecidos else num_itens
    if itens_fornecidos:
        bloco_itens = ("Os itens JÁ FORAM ESCOLHIDOS pelo criador do canal, na ordem em que aparecem no vídeo. "
                       "Use EXATAMENTE estes títulos, sem renomear, trocar ou reordenar:\n" +
                       "\n".join(f"{i + 1}. {t}" for i, t in enumerate(itens_fornecidos)))
    else:
        bloco_itens = (f"Escolha VOCÊ os {n} itens. Devem ser reais, específicos e distintos entre si "
                       f"(nada repetido ou quase igual), e o último item mostrado deve ser o mais forte "
                       f"ou surpreendente — a lista precisa ter um motivo pra ser vista até o fim.")
    sentido = ("CONTAGEM REGRESSIVA: o primeiro item mostrado recebe o número {n} e o último recebe o número 1."
               if ordem == 'regressiva' else
               "ORDEM CRESCENTE: o primeiro item mostrado recebe o número 1 e o último recebe o número {n}.").format(n=n)
    prompt = f"""Você é roteirista de vídeos em LISTA pro YouTube. NÃO escreva prosa ainda — apenas o
esqueleto, em {idioma_conteudo}.

NICHO: {contexto_nicho}
TEMA DA LISTA: "{tema}"
{_nota_publico()}
{bloco_itens}
{sentido}

Regras:
- São exatamente {n} itens.
- O título de cada item é curto (1 a 6 palavras): o NOME do item como aparece num card na tela
  (ex: o modelo da moto, o nome do prato), nunca uma frase.
- Em "cobre", diga que fatos CONCRETOS esse item vai usar (ano, número, lugar, característica
  marcante, curiosidade) e por que ele merece a posição que ocupa na lista.
- "caso_ancoragem" aqui é o CRITÉRIO da lista: o que qualifica um item pra entrar (1 frase).
- A introdução abre com um gancho curto ligado ao tema, deixa claro o critério e promete o que
  vem — sem entregar o item final.
- O desfecho fecha a lista (retoma o critério, comenta o que os itens têm em comum) e termina
  provocando o espectador de um jeito específico deste tema (ex: perguntar qual item ele viveu
  ou qual faltou), não um "e você, o que acha?" genérico.

Retorne APENAS JSON:
{{
  "caso_ancoragem": "o critério da lista",
  "introducao": "o que a introdução deve cobrir",
  "capitulos": [
    {{"titulo": "Nome do Item", "cobre": "fatos concretos + por que está nessa posição"}}
  ],
  "desfecho": "o que o desfecho deve cobrir"
}}
(o array "capitulos" deve ter exatamente {n} itens)"""
    estrutura = _extrair_json(gemini_generate_fn(prompt).text)
    itens = estrutura.get('capitulos', [])
    if itens_fornecidos:  # garante nomes do usuário mesmo se o modelo "melhorar" algum
        for i, titulo in enumerate(itens_fornecidos):
            if i < len(itens):
                itens[i]['titulo'] = titulo
    if len(itens) != n:
        raise ValueError(f"estrutura de lista veio com {len(itens)} itens, esperado {n}")
    return estrutura


def gerar_prosa_capitulos(estrutura_capitulos, contexto_nicho, idioma_conteudo, instrucao_extra,
                           documento_estilo, palavras_alvo, gemini_generate_fn,
                           formato='webdoc', ordem_lista='regressiva'):
    """
    Escreve a prosa de introdução/capítulos/desfecho reaproveitando gerar_prosa (mesmo
    contrato: dict ordenado {{chave: descrição}} -> um bloco de texto por chave), e
    depois anota cada bloco de capítulo com 'titulo_capitulo' + 'inicio_capitulo': True.

    Esses dois metadados extras (que sobrevivem até blocos_com_tempo, porque
    mapear_tempos_para_blocos faz **bloco ao montar o resultado) são o que
    generate_video.py usa pra saber ONDE no vídeo inserir o card preto de transição
    (mudo, sem narração) e trocar a música tema.

    O título do capítulo NÃO entra falado no roteiro (o card é 100% silencioso — só
    texto na tela) — quem garante que o card apareça no momento certo, numa pausa
    de verdade (sem narração por cima), é a montagem do áudio em generate_video.py
    (ver criar_video_webdoc_capitulos), que gera cada capítulo como um arquivo de
    áudio separado e insere um silêncio real do tamanho do card entre eles antes de
    concatenar tudo — não é um truque de texto aqui no roteiro.
    """
    estrutura_prosa = {'introducao': estrutura_capitulos['introducao']}
    for i, cap in enumerate(estrutura_capitulos['capitulos']):
        estrutura_prosa[f'capitulo_{i + 1}'] = cap['cobre']
    estrutura_prosa['desfecho'] = estrutura_capitulos['desfecho']

    # O caso de ancoragem (ver gerar_estrutura_capitulos) precisa estar visível na
    # escrita de TODOS os capítulos, não só da introdução — senão cada capítulo é
    # escrito "às cegas" quanto ao fio condutor do vídeo e a prosa volta a ficar
    # genérica. gerar_prosa já manda TODOS os itens da estrutura numa prompt só (tem
    # contexto de todos ao escrever cada um), então isso só precisa entrar como reforço
    # explícito via instrucao_extra.
    caso_ancoragem = estrutura_capitulos.get('caso_ancoragem', '')
    reforco_densidade = (
        f"CASO DE ANCORAGEM deste vídeo (mantenha ele presente em pelo menos 2 capítulos, "
        f"não só na introdução): {caso_ancoragem}\n"
        f"DENSIDADE DE INFORMAÇÃO: cada parágrafo precisa conter pelo menos UM fato "
        f"específico (número, data, nome de lugar/pessoa/instituição/lei) — nunca uma "
        f"frase que serviria pra qualquer vídeo genérico sobre o nicho. Se uma frase "
        f"pode ser lida sem perder sentido num vídeo sobre outro tema qualquer, ela é "
        f"genérica demais e precisa ser reescrita com um fato concreto no lugar."
    )
    if formato == 'lista':
        n_itens = len(estrutura_capitulos['capitulos'])
        reforco_densidade = (
            f"FORMATO LISTA ({n_itens} itens). CRITÉRIO da lista: {caso_ancoragem}\n"
            f"Cada item é um bloco que abre JÁ NOMEANDO o item e a posição dele (o número por "
            f"extenso, falado) na primeira frase, porque a tela mostra um card com o número e "
            f"o nome — a narração e o card precisam dizer a mesma coisa. Os itens têm "
            f"tamanho parecido entre si (nenhum item é só uma frase). DENSIDADE: cada item traz "
            f"pelo menos dois fatos específicos (ano, número, lugar, característica) — nada "
            f"que serviria pra qualquer item. Não anuncie o item seguinte no fim de cada bloco."
        )
    instrucao_extra_completa = f"{instrucao_extra}\n{reforco_densidade}" if instrucao_extra else reforco_densidade

    blocos = gerar_prosa(estrutura_prosa, contexto_nicho, idioma_conteudo, instrucao_extra_completa,
                          documento_estilo, palavras_alvo, gemini_generate_fn)

    titulos_capitulos = {f'capitulo_{i + 1}': cap['titulo']
                          for i, cap in enumerate(estrutura_capitulos['capitulos'])}
    n_cap = len(estrutura_capitulos['capitulos'])
    for b in blocos:
        titulo_cap = titulos_capitulos.get(b['bloco'])
        b['titulo_capitulo'] = titulo_cap
        b['inicio_capitulo'] = titulo_cap is not None
        if formato == 'lista' and titulo_cap is not None:
            # card com NÚMERO + NOME sobre a mídia do item (sem tela preta nem pausa)
            idx = int(b['bloco'].split('_')[1]) - 1
            b['formato_card'] = 'overlay'
            b['numero_item'] = (n_cap - idx) if ordem_lista == 'regressiva' else (idx + 1)

    return blocos


def gerar_pacote_roteiro_capitulos(tema, contexto_nicho, idioma_conteudo, instrucao_extra,
                                    documento_estilo, tipo_video, gemini_generate_fn,
                                    num_capitulos=3, palavras_alvo=None,
                                    formato='webdoc', itens_lista=None, ordem_lista='regressiva'):
    """
    formato='lista' (ver gerar_estrutura_lista): mesma cadeia, mas os capítulos viram ITENS
    numerados de uma lista, com card de número+nome sobre a mídia em vez de tela preta.

    Modo 'capitulos_webdoc' — formato investigativo em capítulos nomeados, cada um com
    card de transição e música própria. Mesmo contrato de saída de gerar_pacote_roteiro
    (roteiro_texto, roteiro_blocos, titulo, descricao), mais 'capitulos_meta' — lista
    pronta de {{'titulo', 'bloco'}} pra quem não quiser vasculhar roteiro_blocos.
    """
    if palavras_alvo is None and _CFG['chars_min'] and _CFG['chars_max'] and tipo_video != 'short':
        # alvo veio em CARACTERES (config) — converte pra palavras só pro prompt inicial;
        # o tamanho de verdade é medido e corrigido em ajustar_tamanho_por_caracteres()
        palavras_alvo = round(((_CFG['chars_min'] + _CFG['chars_max']) / 2) / _CFG['chars_por_palavra'])
    if palavras_alvo is None:
        # Webdoc investigativo real (tipo Elementar) fica na faixa de 12-20min — a ~140
        # palavras/minuto de narração falada, isso pede uns 1700-2400 palavras de roteiro.
        # 900 (valor antigo) dava só uns 4min de vídeo, curto demais pro formato.
        palavras_alvo = 180 if tipo_video == 'short' else 1800

    try:
        print("  🧱 Estágio 1/4 — estrutura em capítulos...")
        if formato == 'lista':
            estrutura = _com_retry_estagio(
                "Estágio 1/4 (estrutura de lista)", gerar_estrutura_lista,
                tema, contexto_nicho, idioma_conteudo, gemini_generate_fn,
                num_capitulos, itens_lista, ordem_lista)
        else:
            estrutura = _com_retry_estagio(
                "Estágio 1/4 (estrutura)", gerar_estrutura_capitulos,
                tema, contexto_nicho, idioma_conteudo, gemini_generate_fn, num_capitulos)

        print("  ✍️ Estágio 2/4 — escrita...")
        blocos = _com_retry_estagio(
            "Estágio 2/4 (escrita)", gerar_prosa_capitulos,
            estrutura, contexto_nicho, idioma_conteudo, instrucao_extra,
            documento_estilo, palavras_alvo, gemini_generate_fn,
            formato, ordem_lista)

        print("  🔍 Estágio 3/4 — crítica adversarial...")
        blocos = _com_retry_estagio(
            "Estágio 3/4 (crítica)", criticar_e_reescrever,
            blocos, idioma_conteudo, gemini_generate_fn)
        # criticar_e_reescrever reescreve 'texto', mas 'titulo_capitulo'/'inicio_capitulo'
        # são chaves separadas no dict do bloco — sobrevivem à reescrita sem precisar
        # de nenhum reforço aqui (diferente de quando o título entrava embutido no
        # próprio texto falado, que já não é mais o caso).

        if _CFG['chars_min'] and _CFG['chars_max']:
            blocos = _com_retry_estagio(
                "Ajuste de tamanho", ajustar_tamanho_por_caracteres,
                blocos, idioma_conteudo, gemini_generate_fn)

        # Passada final determinística: garante zero dígito/abreviação no texto que vai
        # pro TTS, mesmo que o Gemini tenha deixado algum escapar nas etapas acima.
        for b in blocos:
            b['texto'] = _pos_processar_texto(b['texto'])
            if _CFG['idioma_numeros'] and contem_digitos(b['texto']):
                print(f"  ⚠️ Bloco '{b['bloco']}' ainda tem dígito após a normalização — revise.")

        tema_meta = tema
        if formato == 'lista':
            n_it = len(estrutura['capitulos'])
            tema_meta = (f"{tema} [FORMATO: vídeo de LISTA com {n_it} itens — o título deve conter "
                         f"o número {n_it} e dizer o que a lista traz]")
        print("  🏷️ Estágio 4/4 — título e descrição...")
        titulo = _com_retry_estagio(
            "Estágio 4/4 (título)", gerar_titulo_final_simples,
            tema_meta, estrutura, idioma_conteudo, gemini_generate_fn)
        descricao = _com_retry_estagio(
            "Estágio 4/4 (descrição)", gerar_descricao_final_simples,
            tema_meta, estrutura, idioma_conteudo, gemini_generate_fn)

        roteiro_texto = " ".join(b['texto'] for b in blocos)
        capitulos_meta = [{'titulo': b['titulo_capitulo'], 'bloco': b['bloco']}
                           for b in blocos if b.get('inicio_capitulo')]

        return {
            'roteiro_texto': roteiro_texto,
            'roteiro_blocos': blocos,
            'titulo': titulo,
            'descricao': descricao,
            'tese': None,
            'modo': 'capitulos_webdoc',
            'formato': formato,
            'capitulos_meta': capitulos_meta,
        }
    except Exception as e:
        # BUGFIX: a mensagem de aviso lá em generate_video.py (quando o modo final sai
        # 'fallback_simples') diz "veja o traceback impresso mais acima" — mas até agora
        # nada imprimia esse traceback de fato, só a mensagem curta str(e) abaixo. Sem o
        # traceback, não dá pra saber em qual dos 4 estágios (e em qual linha exata)
        # a cadeia quebrou, o que torna esse tipo de falha praticamente impossível de
        # diagnosticar a partir do log. Agora imprime de verdade.
        print(f"  ⚠️ Cadeia de capítulos falhou ({e}) — usando geração simples de emergência")
        traceback.print_exc()
        prompt_simples = f"""Crie um roteiro de narração investigativo para um vídeo de {contexto_nicho}
sobre "{tema}", em {idioma_conteudo}, ~{palavras_alvo} palavras, tom direto e factual.
Escreva APENAS o roteiro corrido."""
        resposta = gemini_generate_fn(prompt_simples)
        roteiro_texto = re.sub(r'\*+', '', resposta.text).replace('#', '').strip()
        return {
            'roteiro_texto': roteiro_texto,
            'roteiro_blocos': [{'bloco': 'roteiro_completo', 'texto': roteiro_texto}],
            'titulo': tema,
            'descricao': {'abertura_seo': tema, 'corpo': roteiro_texto[:200]},
            'tese': None,
            'modo': 'fallback_simples',
            'capitulos_meta': [],
        }


def _resumo_estrutura_para_prompt(estrutura):
    """
    Normaliza QUALQUER uma das estruturas de roteiro (devocional OU capítulos) num
    resumo textual pros prompts de título/descrição. BUGFIX: antes, título e descrição
    liam direto 'reflexao_central'/'aplicacao_pratica' — chaves que só existem na
    estrutura devocional. Chamado a partir do modo 'capitulos_webdoc' (estrutura
    {{introducao, capitulos, desfecho}}), isso retornava string vazia pros dois campos,
    ou seja, o prompt de título rodava praticamente sem contexto nenhum do vídeo — só
    o tema cru — o que explica títulos/ganchos fracos e genéricos.
    """
    if 'capitulos' in estrutura:
        partes = [f"INTRODUÇÃO (dado de abertura): {estrutura.get('introducao', '')}"]
        for i, cap in enumerate(estrutura.get('capitulos', [])):
            partes.append(f"CAPÍTULO {i + 1} \"{cap.get('titulo', '')}\": {cap.get('cobre', '')}")
        partes.append(f"DESFECHO: {estrutura.get('desfecho', '')}")
        return "\n".join(partes)
    return (f"REFLEXÃO CENTRAL: {estrutura.get('reflexao_central', '')}\n"
            f"APLICAÇÃO PRÁTICA: {estrutura.get('aplicacao_pratica', '')}")


def gerar_titulo_final_simples(tema, estrutura, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Gere 10 variantes de título de vídeo para YouTube, em {idioma_conteudo}.
Cada variante prioriza EXPLICITAMENTE um destes critérios (rotule qual):
- gap_curiosidade: cria uma lacuna de informação sem entregar a resposta
- especificidade: usa um NÚMERO ou detalhe concreto do vídeo (valor monetário na moeda do público, %, quantidade)
- emocao: nomeia o sentimento/estado que o vídeo aborda (indignação, choque, injustiça)

TEMA: {tema}
{_resumo_estrutura_para_prompt(estrutura)}

O título tem que dar pra entender do que o vídeo trata SEM contexto adicional — nunca
um número ou palavra solta sem gancho (ex: "QUATRO BILHÕES" sozinho não diz nada; "A
CIDADE QUE GASTOU 4 BILHÕES E NÃO CONSTRUIU NADA" diz). Pense em títulos de documentário
investigativo: "[Lugar/Coisa específica]: [a virada/problema] | [enquadramento]" — como
"RS: A Fábrica de Cidades Inúteis?" ou "Por que Condomínios Estão Tão Caros?".

Para cada variante, responda também: esse título serviria pra QUALQUER vídeo desse
nicho, ou só faz sentido pra ESTE vídeo específico?

Retorne APENAS JSON:
{{
  "titulos": [
    {{"texto": "...", "criterio": "...", "exclusivo_deste_video": true}}
  ]
}}"""

    resposta = gemini_generate_fn(prompt)
    dados = _extrair_json(resposta.text)
    candidatos = [t for t in dados.get('titulos', []) if t.get('exclusivo_deste_video')]
    if not candidatos:
        candidatos = dados.get('titulos', [])
    if not candidatos:
        return tema[:60]
    return candidatos[0]['texto']


def gerar_descricao_final_simples(tema, estrutura, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Escreva a descrição do vídeo em duas partes, em {idioma_conteudo}.

TEMA: {tema}
{_resumo_estrutura_para_prompt(estrutura)}

1. abertura_seo (até 150 caracteres): termos de busca reais que alguém digitaria sobre
   esse tema — não invente termos.
2. corpo: o que o espectador vai encontrar no vídeo, sem entregar a reflexão inteira.

Retorne APENAS JSON: {{"abertura_seo": "...", "corpo": "..."}}"""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


# ============================================================
# ESTÁGIO 4 — CRÍTICA ADVERSARIAL + REESCRITA CIRÚRGICA
# ============================================================

def criticar_e_reescrever(blocos, idioma_conteudo, gemini_generate_fn):
    """
    Chamada separada da escrita de propósito: crítica e geração são tarefas cognitivas
    diferentes, e o mesmo prompt que gera o clichê raramente enxerga o próprio clichê.
    Só reescreve os blocos marcados com problema — nunca o roteiro inteiro de novo.
    """
    roteiro_numerado = "\n".join(f"[{i}] ({b['bloco']}): {b['texto']}" for i, b in enumerate(blocos))

    prompt = f"""Você é um editor implacável. Sua ÚNICA função é achar defeito, nunca elogiar
nem reescrever ainda.

ROTEIRO (em {idioma_conteudo}, um bloco numerado por linha):
{roteiro_numerado}

Para cada bloco com problema, marque o tipo:
- clichê: frase que poderia estar em qualquer vídeo do nicho
- ritmo_quebrado: frase longa demais ou estrutura repetitiva
- redundancia: informação repetida sem necessidade
- generico: frase sem NENHUM fato específico (número, data, nome de lugar/pessoa/
  instituição/lei) — se a frase pode ser lida sem perder sentido num vídeo sobre
  qualquer outro tema do mesmo nicho, ela é genérica demais

Retorne APENAS JSON, lista de problemas (vazio se não houver nenhum):
{{"problemas": [{{"indice": 0, "tipo": "cliche", "motivo": "..."}}]}}"""

    resposta = gemini_generate_fn(prompt)
    problemas = _extrair_json(resposta.text).get('problemas', [])

    if not problemas:
        return blocos

    indices_com_problema = sorted({p['indice'] for p in problemas if 0 <= p['indice'] < len(blocos)})
    if not indices_com_problema:
        return blocos

    trechos = "\n".join(
        f"[{i}] ({blocos[i]['bloco']}): {blocos[i]['texto']}\n"
        f"    problema: {[p['motivo'] for p in problemas if p['indice'] == i]}"
        for i in indices_com_problema
    )

    prompt_reescrita = f"""Reescreva APENAS os blocos abaixo, corrigindo o problema apontado.
Mantenha o mesmo sentido e o mesmo tamanho aproximado. Escreva em {idioma_conteudo}.

{trechos}

IMPORTANTE — formato da resposta: "texto_novo" deve ser SÓ o texto puro que vai ser
narrado, exatamente como sairia numa legenda. NUNCA repita o prefixo "[n] (nome_do_bloco):"
usado acima pra identificar os trechos — isso é só uma referência, não faz parte do roteiro.
Não inclua colchetes, números de índice, nem o nome do bloco entre parênteses.

IMPORTANTE — texto vai pra um TTS que lê tudo de forma LITERAL: se o texto original
menciona um país/órgão/instituição por extenso (ex: "Estados Unidos"), o texto_novo
tem que MANTER por extenso. Nunca troque isso por sigla, abreviação, ou por uma letra/
código curto pra "variar" o texto ou evitar repetição — se precisar variar, use uma
expressão equivalente por extenso ("o país", "a potência americana"), nunca uma sigla.{_regra_numeros()}

Retorne APENAS JSON: {{"reescritas": [{{"indice": 0, "texto_novo": "..."}}]}}"""

    resposta2 = gemini_generate_fn(prompt_reescrita)
    reescritas = _extrair_json(resposta2.text).get('reescritas', [])

    # Defesa extra: mesmo com a instrução acima, o modelo às vezes ainda ecoa o
    # prefixo "[n] (bloco): " de volta no texto_novo — se isso acontecer, o prefixo
    # vazaria pra narração E pra legenda (roteiro_texto alimenta as duas). Removemos
    # qualquer prefixo nesse formato antes de aceitar o texto.
    padrao_prefixo = re.compile(r'^\s*\[\d+\]\s*\([^)]*\)\s*:\s*')

    for r in reescritas:
        i = r.get('indice')
        if isinstance(i, int) and 0 <= i < len(blocos) and r.get('texto_novo'):
            texto_novo = padrao_prefixo.sub('', r['texto_novo'].strip())
            blocos[i]['texto'] = _pos_processar_texto(sanitizar_direcoes_de_producao(texto_novo.strip()))

    return blocos


# ============================================================
# ESTÁGIO 5 — TÍTULO E DESCRIÇÃO (pipeline próprio)
# ============================================================

def gerar_titulo_final(tese_dict, estrutura, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Gere 10 variantes de título de vídeo para YouTube, em {idioma_conteudo}.
Cada variante prioriza EXPLICITAMENTE um destes critérios (rotule qual):
- gap_curiosidade: cria uma lacuna de informação sem entregar a resposta
- especificidade: usa um detalhe concreto do vídeo
- contraste: usa a objeção/crença comum que a tese contraria

TESE: {tese_dict['tese']}
OBJEÇÃO: {estrutura['objecao']}
GANCHO: {estrutura['gancho']}

Para cada variante, responda também: esse título serviria pra QUALQUER vídeo desse nicho,
ou só faz sentido pra ESTE vídeo específico?

Retorne APENAS JSON:
{{
  "titulos": [
    {{"texto": "...", "criterio": "...", "exclusivo_deste_video": true}}
  ]
}}"""

    resposta = gemini_generate_fn(prompt)
    dados = _extrair_json(resposta.text)
    candidatos = [t for t in dados.get('titulos', []) if t.get('exclusivo_deste_video')]

    if not candidatos:
        candidatos = dados.get('titulos', [])
    if not candidatos:
        return tese_dict['tese'][:60]

    return candidatos[0]['texto']


def gerar_descricao_final(tese_dict, estrutura, idioma_conteudo, gemini_generate_fn):
    prompt = f"""Escreva a descrição do vídeo em duas partes, em {idioma_conteudo}.

TESE: {tese_dict['tese']}
OBJEÇÃO: {estrutura['objecao']}
IMPLICAÇÃO: {estrutura['implicacao']}

1. abertura_seo (até 150 caracteres): termos de busca reais que alguém digitaria sobre
   esse tema — não invente termos.
2. corpo: o que o espectador vai aprender e por que a tese é contestável, SEM entregar
   a conclusão do vídeo.

Retorne APENAS JSON: {{"abertura_seo": "...", "corpo": "..."}}"""

    resposta = gemini_generate_fn(prompt)
    return _extrair_json(resposta.text)


# ============================================================
# ORQUESTRAÇÃO — chamada única a partir do generate_video.py
# ============================================================

def gerar_pacote_roteiro(tema, contexto_nicho, idioma_conteudo, instrucao_extra,
                          documento_estilo, tipo_video, gemini_generate_fn,
                          modo_roteiro='cadeia_completa', num_capitulos=3, palavras_alvo_webdoc=None,
                          formato='webdoc', itens_lista=None, ordem_lista='regressiva'):
    """
    Roda a cadeia completa (modo_roteiro='cadeia_completa', padrão), o modo simples
    sem tese/objeção (modo_roteiro='simples'), ou o modo webdoc em capítulos nomeados
    (modo_roteiro='capitulos_webdoc' — ver gerar_pacote_roteiro_capitulos). Os três
    modos devolvem exatamente o mesmo formato de saída (roteiro_texto, roteiro_blocos,
    titulo, descricao), então nada em generate_video.py precisa saber qual modo rodou —
    inclusive a Fase 2 de produção visual (B-roll por bloco, destaque, SFX) funciona
    igual nos três, porque só depende de roteiro_blocos existir, não de COMO ele foi
    gerado. Se qualquer estágio falhar, cai pra um roteiro simples de emergência (uma
    chamada só) em vez de quebrar o workflow inteiro — mesmo espírito de fallback que
    já existe em criar_audio().
    """
    palavras_alvo = 180 if tipo_video == 'short' else 650

    if modo_roteiro == 'capitulos_webdoc':
        return gerar_pacote_roteiro_capitulos(
            tema, contexto_nicho, idioma_conteudo, instrucao_extra, documento_estilo,
            tipo_video, gemini_generate_fn, num_capitulos=num_capitulos,
            palavras_alvo=palavras_alvo_webdoc,
            formato=formato, itens_lista=itens_lista, ordem_lista=ordem_lista
        )

    try:
        if modo_roteiro == 'simples':
            print("  🧱 Estágio 1/4 — estrutura devocional (sem tese/objeção)...")
            estrutura = _com_retry_estagio(
                "Estágio 1/4 (estrutura)", gerar_estrutura_devocional,
                tema, contexto_nicho, idioma_conteudo, gemini_generate_fn)

            print("  ✍️ Estágio 2/4 — escrita...")
            blocos = _com_retry_estagio(
                "Estágio 2/4 (escrita)", gerar_prosa,
                estrutura, contexto_nicho, idioma_conteudo, instrucao_extra,
                documento_estilo, palavras_alvo, gemini_generate_fn)

            print("  🔍 Estágio 3/4 — crítica adversarial...")
            blocos = _com_retry_estagio(
                "Estágio 3/4 (crítica)", criticar_e_reescrever,
                blocos, idioma_conteudo, gemini_generate_fn)

            print("  🏷️ Estágio 4/4 — título e descrição...")
            titulo = _com_retry_estagio(
                "Estágio 4/4 (título)", gerar_titulo_final_simples,
                tema, estrutura, idioma_conteudo, gemini_generate_fn)
            descricao = _com_retry_estagio(
                "Estágio 4/4 (descrição)", gerar_descricao_final_simples,
                tema, estrutura, idioma_conteudo, gemini_generate_fn)

            roteiro_texto = " ".join(b['texto'] for b in blocos)
            return {
                'roteiro_texto': roteiro_texto,
                'roteiro_blocos': blocos,
                'titulo': titulo,
                'descricao': descricao,
                'tese': None,
                'modo': 'simples',
            }

        print("  🎯 Estágio 1/5 — tese...")
        tese_dict = _com_retry_estagio(
            "Estágio 1/5 (tese)", gerar_tese,
            tema, contexto_nicho, idioma_conteudo, gemini_generate_fn)
        if tese_dict.get('reprovar'):
            raise ValueError(f"Tese reprovada pelo próprio modelo: {tese_dict.get('tese')}")
        print(f"     tese: {tese_dict['tese']}")

        print("  🧱 Estágio 2/5 — estrutura argumentativa...")
        estrutura = _com_retry_estagio(
            "Estágio 2/5 (estrutura)", gerar_estrutura,
            tese_dict, contexto_nicho, idioma_conteudo, gemini_generate_fn)

        print("  ✍️ Estágio 3/5 — escrita...")
        blocos = _com_retry_estagio(
            "Estágio 3/5 (escrita)", gerar_prosa,
            estrutura, contexto_nicho, idioma_conteudo, instrucao_extra,
            documento_estilo, palavras_alvo, gemini_generate_fn)

        print("  🔍 Estágio 4/5 — crítica adversarial...")
        blocos = _com_retry_estagio(
            "Estágio 4/5 (crítica)", criticar_e_reescrever,
            blocos, idioma_conteudo, gemini_generate_fn)

        print("  🏷️ Estágio 5/5 — título e descrição...")
        titulo = _com_retry_estagio(
            "Estágio 5/5 (título)", gerar_titulo_final,
            tese_dict, estrutura, idioma_conteudo, gemini_generate_fn)
        descricao = _com_retry_estagio(
            "Estágio 5/5 (descrição)", gerar_descricao_final,
            tese_dict, estrutura, idioma_conteudo, gemini_generate_fn)

        roteiro_texto = " ".join(b['texto'] for b in blocos)

        return {
            'roteiro_texto': roteiro_texto,
            'roteiro_blocos': blocos,
            'titulo': titulo,
            'descricao': descricao,
            'tese': tese_dict['tese'],
            'modo': 'cadeia_completa',
        }

    except Exception as e:
        print(f"  ⚠️ Cadeia de roteiro falhou ({e}) — usando geração simples de emergência")
        traceback.print_exc()
        prompt_simples = f"""Crie um roteiro de narração para um vídeo de {contexto_nicho} sobre "{tema}",
em {idioma_conteudo}, ~{palavras_alvo} palavras, tom direto. Escreva APENAS o roteiro corrido."""
        resposta = gemini_generate_fn(prompt_simples)
        roteiro_texto = re.sub(r'\*+', '', resposta.text).replace('#', '').strip()

        return {
            'roteiro_texto': roteiro_texto,
            'roteiro_blocos': [{'bloco': 'roteiro_completo', 'texto': roteiro_texto}],
            'titulo': tema,
            'descricao': {'abertura_seo': tema, 'corpo': roteiro_texto[:200]},
            'tese': None,
            'modo': 'fallback_simples',
        }
