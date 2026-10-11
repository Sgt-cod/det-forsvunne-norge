"""
telegram_review.py
-------------------
Camada de aprovação humana via Telegram, opt-in via config.json ('telegram_review' /
'selecao_tema_telegram'). Não usa nenhuma lib de bot (python-telegram-bot etc.) — só
'requests' puro contra a Bot API do Telegram, no mesmo estilo do resto do pipeline
(pesquisar_videos_pexels, buscar_imagem_wikimedia...), pra não adicionar dependência
nova ao requirements.txt.

SECRETS NECESSÁRIOS (env vars / GitHub Actions secrets):
  TELEGRAM_BOT_TOKEN — token do bot, gerado pelo @BotFather no Telegram
  TELEGRAM_CHAT_ID   — chat_id de destino. Pra descobrir o seu: mande QUALQUER
                        mensagem pro bot recém-criado, depois abra no navegador
                        https://api.telegram.org/bot<SEU_TOKEN>/getUpdates — o campo
                        "message":{"chat":{"id": ...}} é o número que você quer.

Se qualquer uma das duas variáveis não estiver setada, ATIVA_TELEGRAM fica False e
todo o resto deste módulo vira no-op silencioso — nada quebra pra quem não configurou
(o pipeline segue 100% automático, igual antes desse módulo existir).

FLUXO DE REVISÃO DE MÍDIA (revisar_midia_pipeline):
  pra cada clipe, na ordem: manda a mídia + o trecho do roteiro correspondente +
  [Aprovar]/[Recusar]
    - Aprovar → segue pro próximo clipe
    - Recusar → manda [Cancelar workflow]/[Enviar mídia]
        - Cancelar → levanta WorkflowCanceladoPeloUsuario (main() captura e encerra
          sem publicar nada)
        - Enviar mídia → espera a PRÓXIMA mensagem: se for um link do Pexels
          (pexels.com/video/... ou /photo/...), baixa aquele item específico pela API;
          se for uma foto/vídeo/documento enviado direto do celular, baixa o arquivo do
          Telegram — em ambos os casos substitui o 'path' do clipe (o timing do corte
          não muda) e segue pro próximo

FLUXO DE SELEÇÃO DE TEMA (escolher_tema_telegram):
  manda uma pergunta com botão [Nada a sugerir]; se a resposta vier como TEXTO, isso
  vira o tema/direcionamento do próximo roteiro; se vier o botão (ou estourar o
  timeout), retorna None e quem chamou cai pra escolha automática de sempre
  (escolher_tema_reflexao(), dentro de generate_video.py).

Ambos os fluxos são SÍNCRONOS/bloqueantes (long-polling no getUpdates) — aceitável
porque o pipeline já roda como um script linear (GitHub Actions ou local), mas atenção
ao timeout do job/runner: se o timeout de resposta configurado for maior que o timeout
do job, o job morre no meio da espera. Ajuste 'timeout_resposta_min' no config.json de
acordo com o timeout do seu runner.
"""

import os
import re
import time
import json

import hashlib
import itertools
import subprocess
import requests
from PIL import Image, ImageOps
from rede_utils import com_watchdog

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
ATIVA_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

PEXELS_API_KEY = os.environ.get('PEXELS_API_KEY')

_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None


class WorkflowCanceladoPeloUsuario(Exception):
    """Levantada quando o usuário aperta 'Cancelar workflow' no Telegram (ou não
    responde ao cancelamento a tempo). main() deve capturar isso especificamente e
    encerrar SEM publicar/subir nada — não é uma falha do pipeline, é uma decisão."""
    pass


def _config():
    try:
        with open(os.environ.get('CONFIG_FILE', 'config.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _timeout_min(chave, padrao):
    return int(_config().get(chave, {}).get('timeout_resposta_min', padrao))


# ============================================================
# Primitivas da Bot API
# ============================================================

def _chamar(metodo, **params):
    resp = _requisitar_com_retry('post', f"{_API_BASE}/{metodo}", data=params, timeout=30)
    dados = resp.json()
    if not dados.get('ok'):
        raise RuntimeError(f"Telegram API '{metodo}' falhou: {dados}")
    return dados['result']


def _requisitar_com_retry(verbo, url, tentativas=3, espera_s=3, **kwargs):
    """
    BUGFIX (Telegram parou de interagir no meio do workflow): getUpdates é uma conexão
    de long-polling — só UMA pode estar "ativa" por bot de cada vez. Se uma run anterior
    for cancelada manualmente (ex: no GitHub Actions) enquanto uma requisição de
    getUpdates está em aberto, o Telegram pode devolver 409 Conflict ("terminated by
    other getUpdates request") pras primeiras chamadas da run seguinte, até a conexão
    antiga expirar de vez do lado do servidor do Telegram. Sem isso, uma única 409
    (ou qualquer erro de rede transitório) subia sem tratamento e travava a interação
    pro resto do workflow. Agora tenta de novo, com espera progressiva, antes de desistir.
    """
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.request(verbo, url, **kwargs)
            if resp.status_code == 409:
                raise requests.exceptions.HTTPError(f"409 Conflict em {url}", response=resp)
            resp.raise_for_status()
            return resp
        except Exception as e:
            ultimo_erro = e
            if tentativa < tentativas:
                print(f"    ⚠️ Telegram API falhou (tentativa {tentativa}/{tentativas}: {e}) "
                      f"— tentando de novo em {espera_s}s...")
                time.sleep(espera_s * tentativa)
    raise ultimo_erro


def _enviar_arquivo(metodo, campo_arquivo, caminho, **params):
    # lê o arquivo pra memória ANTES de tentar — se _requisitar_com_retry precisar
    # tentar de novo (ex: depois de um 409), reabrir o arquivo a cada tentativa
    # evitaria o bug de mandar corpo vazio na 2ª tentativa (stream já consumido)
    with open(caminho, 'rb') as f:
        conteudo = f.read()
    nome_arquivo = os.path.basename(caminho)
    resp = _requisitar_com_retry('post', f"{_API_BASE}/{metodo}", data=params,
                                  files={campo_arquivo: (nome_arquivo, conteudo)}, timeout=120)
    dados = resp.json()
    if not dados.get('ok'):
        raise RuntimeError(f"Telegram API '{metodo}' falhou: {dados}")
    return dados['result']


def _teclado(botoes):
    """botoes: lista de (texto, callback_data) → UMA linha; ou lista de listas → várias
    linhas. Se callback_data começa com 'url:', vira botão de LINK (abre no navegador,
    não gera callback pro bot)."""
    linhas = botoes if (botoes and isinstance(botoes[0], list)) else [botoes]
    def _b(t, d):
        return {'text': t, 'url': d[4:]} if d.startswith('url:') else {'text': t, 'callback_data': d}
    return json.dumps({'inline_keyboard': [[_b(t, d) for t, d in linha] for linha in linhas]})


def enviar_texto(texto, botoes=None):
    """botoes: ver _teclado()."""
    params = {'chat_id': TELEGRAM_CHAT_ID, 'text': texto}
    if botoes:
        params['reply_markup'] = _teclado(botoes)
    return _chamar('sendMessage', **params)


# ============================================================
# Texto paralelo em PT-BR (canal em outro idioma)
# ============================================================
# Quando o canal narra em outro idioma (ex: norueguês) e quem opera não lê esse idioma, toda
# mensagem que traz texto do roteiro vem com a tradução em PT-BR logo abaixo. O tradutor é
# injetado por generate_video.py (usa o mesmo Gemini do resto do pipeline) pra este módulo
# continuar sem dependência do Gemini. Sem tradutor configurado, tudo segue como antes.

_TRADUTOR = None
_CACHE_TRADUCAO = {}   # (texto, com_sugestao) -> {'pt','sugestao'}; o mesmo texto de segmento
                       # é mostrado em mais de uma etapa (destaques + áudio): 1 chamada só


def configurar_tradutor(fn):
    """fn(textos: list[str], com_sugestao: bool) -> list[{'pt': str, 'sugestao': str|None}]"""
    global _TRADUTOR
    _TRADUTOR = fn


def _traduzir(textos, com_sugestao=False):
    """Sempre devolve uma lista do mesmo tamanho de 'textos' — itens None se a tradução
    falhar (a revisão segue só com o texto original em vez de travar)."""
    vazio = [{'pt': None, 'sugestao': None} for _ in textos]
    if _TRADUTOR is None or not textos:
        return vazio
    try:
        faltam = [t for t in textos if (t, com_sugestao) not in _CACHE_TRADUCAO]
        if faltam:
            novos = _TRADUTOR(faltam, com_sugestao)
            if len(novos) != len(faltam):
                raise ValueError(f"tradutor devolveu {len(novos)} itens pra {len(faltam)} textos")
            for t, n in zip(faltam, novos):
                _CACHE_TRADUCAO[(t, com_sugestao)] = n
        return [_CACHE_TRADUCAO[(t, com_sugestao)] for t in textos]
    except Exception as e:
        print(f"  ⚠️ Tradução PT-BR indisponível ({e}) — mandando só o texto original")
        return vazio


def _url_busca_pexels(termo):
    """Link de busca de VÍDEOS no site do Pexels já com os filtros do config.json
    ('pexels_busca_telegram': {"ativo": true, "orientation": "landscape", "min_duration": 10}).
    Os filtros vão na query string; se o site ignorar algum, a busca ainda abre normalmente."""
    cfg = _config().get('pexels_busca_telegram', {})
    if not cfg.get('ativo', False) or not termo:
        return None
    from urllib.parse import quote, urlencode
    filtros = {}
    if cfg.get('orientation', 'landscape'):
        filtros['orientation'] = cfg.get('orientation', 'landscape')
    if cfg.get('min_duration', 10):
        filtros['min_duration'] = cfg.get('min_duration', 10)
    url = f"https://www.pexels.com/search/videos/{quote(termo.strip(), safe='')}/"
    return url + ('?' + urlencode(filtros) if filtros else '')


def _quebrar_texto(texto, limite=3800):
    """Quebra em pedaços <= limite (o Telegram recusa mensagem > 4096), preferindo fim de
    parágrafo/frase pra nunca cortar uma palavra no meio."""
    texto = texto.strip()
    pedacos = []
    while len(texto) > limite:
        corte = max(texto.rfind('\n', 0, limite), texto.rfind('. ', 0, limite) + 1,
                    texto.rfind('? ', 0, limite) + 1, texto.rfind('! ', 0, limite) + 1)
        if corte <= limite // 2:
            corte = limite
        pedacos.append(texto[:corte].strip())
        texto = texto[corte:].strip()
    if texto:
        pedacos.append(texto)
    return pedacos


def enviar_texto_longo(texto, cabecalho=None, botoes=None):
    """enviar_texto() que aguenta textos acima de 4096 caracteres. 'botoes' (se houver)
    vai só na ÚLTIMA mensagem."""
    pedacos = _quebrar_texto(texto)
    for i, pedaco in enumerate(pedacos):
        topo = f"{cabecalho}\n\n" if (cabecalho and i == 0) else ""
        ultimo = (i == len(pedacos) - 1)
        enviar_texto(topo + pedaco, botoes=botoes if ultimo else None)


def enviar_texto_com_traducao(texto, cabecalho_original, cabecalho_pt="🇧🇷 Tradução (só pra você entender)"):
    """Manda o texto original (limpo, fácil de copiar) e, em mensagem separada, a tradução."""
    enviar_texto_longo(texto, cabecalho=cabecalho_original)
    pt = _traduzir([texto])[0]['pt']
    if pt:
        enviar_texto_longo(pt, cabecalho=cabecalho_pt)
    return pt


def enviar_midia(caminho, legenda, botoes=None):
    """Envia foto (jpg/png) ou vídeo (mp4/mov) com legenda + botões inline. Formato não
    reconhecido cai pra mensagem de texto (nunca derruba a revisão por causa disso)."""
    params = {'chat_id': TELEGRAM_CHAT_ID, 'caption': legenda[:1024]}
    if botoes:
        params['reply_markup'] = _teclado(botoes)
    ext = os.path.splitext(caminho)[1].lower()
    try:
        if ext in ('.jpg', '.jpeg', '.png'):
            return _enviar_arquivo('sendPhoto', 'photo', caminho, **params)
        elif ext in ('.mp4', '.mov'):
            return _enviar_arquivo('sendVideo', 'video', caminho, **params)
        else:
            return enviar_texto(f"{legenda}\n\n(mídia em formato sem preview: {ext})", botoes)
    except Exception as e:
        print(f"  ⚠️ Falha ao enviar mídia pro Telegram ({e}) — enviando só o texto")
        return enviar_texto(legenda, botoes)


_offset_updates = None
_offset_inicializado = False

# BUGFIX (mídia trocada/perdida/vazando pra outro segmento): antes, cada função de
# espera (aguardar_callback / aguardar_midia_ou_texto) só reconhecia o TIPO de resposta
# que ela mesma esperava naquele instante — um clique de botão OU uma mídia, nunca os
# dois, e só se chegasse durante a janela exata em que aquela função estava rodando.
# Qualquer mensagem que chegasse "fora de hora" (ex: usuário manda a foto de
# substituição antes de clicar Recusar→Enviar mídia, ou manda o próximo link enquanto
# o bot ainda está processando o clipe anterior) era descartada pra sempre — o
# getUpdates do Telegram não devolve a mesma mensagem duas vezes depois que o offset
# passa por ela. Isso é o que causava clipe pulado, ordem trocada, e mídia de um
# segmento vazando pro próximo (a mensagem "atrasada" ficava pendente e era capturada
# pelo primeiro aguardar_... do segmento SEGUINTE).
#
# A correção: TODA atualização que chega é imediatamente classificada e guardada numa
# fila (callback ou mensagem) — nunca descartada. Cada função de espera primeiro olha
# se já tem algo pendente na fila certa antes de fazer long-polling por algo novo. Isso
# também permite mandar tudo em sequência sem esperar o bot perguntar de novo a cada
# clipe (que é como a maioria das pessoas natural mente usa isso).
_fila_callbacks = []
_fila_mensagens = []


def _pasta_downloads_padrao():
    d = os.path.join('assets', 'telegram_review')
    os.makedirs(d, exist_ok=True)
    return d


def limpar_filas_pendentes():
    """Descarta (com aviso) qualquer callback/mensagem que tenha sobrado sem ser
    consumido no passo anterior. Chamado no INÍCIO da revisão de CADA segmento — é o
    que impede uma resposta atrasada do segmento anterior de vazar pro de agora."""
    global _fila_callbacks, _fila_mensagens
    if _fila_callbacks or _fila_mensagens:
        aviso = (f"🧹 Descartando {len(_fila_callbacks)} clique(s) e "
                 f"{len(_fila_mensagens)} mensagem(ns) que sobraram sem uso do "
                 f"segmento anterior (chegaram atrasadas — se era uma mídia pra um "
                 f"clipe específico, manda de novo quando eu pedir).")
        print(f"  {aviso}")
        try:
            enviar_texto(aviso)
        except Exception:
            pass
    _fila_callbacks = []
    _fila_mensagens = []


def _descartar_atualizacoes_antigas():
    """Roda uma vez, na primeira chamada de qualquer fluxo — drena updates pendentes
    de ANTES desta execução do pipeline (ex: um clique perdido de uma rodada anterior)
    sem processá-los, só pra não confundir a revisão de agora com lixo de outra vez."""
    global _offset_updates, _offset_inicializado
    if _offset_inicializado:
        return
    _offset_inicializado = True
    try:
        resp = _requisitar_com_retry('get', f"{_API_BASE}/getUpdates",
                                      params={'timeout': 0}, timeout=15)
        updates = resp.json().get('result', [])
        if updates:
            _offset_updates = updates[-1]['update_id'] + 1
    except Exception as e:
        print(f"    ⚠️ Não consegui limpar updates antigos do Telegram ({e}) — seguindo mesmo assim")


def _proximos_updates(timeout_long_poll=25):
    """Busca updates novos do Telegram e devolve a lista crua — usado só por
    escolher_tema_telegram/escolher_destaques_telegram, que têm laços próprios (não
    passam pelas filas _fila_callbacks/_fila_mensagens porque rodam ANTES/fora do
    laço de revisão de clipe a clipe, sem risco de mistura entre clipes).

    BUGFIX: antes, um erro aqui (ex: 409 Conflict — ver _requisitar_com_retry) subia
    sem tratamento e travava a interação pro resto do workflow. Agora tenta de novo
    algumas vezes; se mesmo assim falhar, devolve lista vazia (o loop de quem chamou
    simplesmente tenta de novo no próximo ciclo) em vez de derrubar o processo inteiro.
    """
    global _offset_updates
    _descartar_atualizacoes_antigas()
    params = {'timeout': timeout_long_poll}
    if _offset_updates is not None:
        params['offset'] = _offset_updates
    try:
        resp = _requisitar_com_retry('get', f"{_API_BASE}/getUpdates", params=params,
                                      timeout=timeout_long_poll + 10)
    except Exception as e:
        print(f"    ⚠️ getUpdates do Telegram falhou repetidamente ({e}) — tentando de novo...")
        return []
    dados = resp.json()
    if not dados.get('ok'):
        return []
    updates = dados['result']
    if updates:
        _offset_updates = updates[-1]['update_id'] + 1
    return updates


def _classificar_update(upd):
    """Devolve ('callback', data) ou ('mensagem', dict) ou None — NUNCA descarta um
    update reconhecível, só ignora update de outro chat/tipo que não interessa (ex:
    edited_message, my_chat_member)."""
    cq = upd.get('callback_query')
    if cq and str(cq.get('message', {}).get('chat', {}).get('id')) == str(TELEGRAM_CHAT_ID):
        try:
            _chamar('answerCallbackQuery', callback_query_id=cq['id'])
        except Exception:
            pass
        return ('callback', cq.get('data'))

    msg = upd.get('message')
    if not msg or str(msg.get('chat', {}).get('id')) != str(TELEGRAM_CHAT_ID):
        return None

    if msg.get('photo'):
        maior = max(msg['photo'], key=lambda p: p.get('file_size', 0))
        caminho = _baixar_arquivo_telegram(maior['file_id'], _pasta_downloads_padrao(), '.jpg')
        return ('mensagem', {'tipo': 'foto', 'caminho': caminho, 'texto': None})
    if msg.get('video'):
        caminho = _baixar_arquivo_telegram(msg['video']['file_id'], _pasta_downloads_padrao(), '.mp4')
        return ('mensagem', {'tipo': 'video', 'caminho': caminho, 'texto': None})
    if msg.get('document'):
        nome = msg['document'].get('file_name', 'arquivo')
        ext = os.path.splitext(nome)[1].lower() or '.bin'
        caminho = _baixar_arquivo_telegram(msg['document']['file_id'], _pasta_downloads_padrao(), ext)
        return ('mensagem', {'tipo': 'documento', 'caminho': caminho, 'texto': None})
    if msg.get('voice'):
        # Nota de voz do Telegram: sempre .ogg (Opus). O ffmpeg abre esse formato
        # sem problema, mas convertemos pra mp3 na hora de usar (ver
        # _normalizar_audio_para_segmento), já que o resto do pipeline espera mp3.
        caminho = _baixar_arquivo_telegram(msg['voice']['file_id'], _pasta_downloads_padrao(), '.ogg')
        return ('mensagem', {'tipo': 'audio', 'caminho': caminho, 'texto': None})
    if msg.get('audio'):
        nome = msg['audio'].get('file_name', 'audio')
        ext = os.path.splitext(nome)[1].lower() or '.mp3'
        caminho = _baixar_arquivo_telegram(msg['audio']['file_id'], _pasta_downloads_padrao(), ext)
        return ('mensagem', {'tipo': 'audio', 'caminho': caminho, 'texto': None})
    if msg.get('text'):
        return ('mensagem', {'tipo': 'texto', 'caminho': None, 'texto': msg['text'].strip()})
    return None


def _drenar_para_filas(timeout_long_poll=25):
    """Busca updates novos e empilha CADA UM na fila certa (_fila_callbacks ou
    _fila_mensagens) — a peça central do bugfix: nada é descartado só porque quem
    chamou não é quem esperava por aquele tipo específico de resposta."""
    for upd in _proximos_updates(timeout_long_poll=timeout_long_poll):
        classificado = _classificar_update(upd)
        if not classificado:
            continue
        tipo, valor = classificado
        (_fila_callbacks if tipo == 'callback' else _fila_mensagens).append(valor)


def aguardar_callback(timeout_s=1800):
    """Espera o usuário apertar um botão inline. Olha a fila ANTES de fazer
    long-polling — se o clique já tinha chegado (ex: enquanto processava o clipe
    anterior), pega ele na hora em vez de esperar de novo. Retorna None no timeout."""
    limite = time.time() + timeout_s
    while time.time() < limite:
        if _fila_callbacks:
            return _fila_callbacks.pop(0)
        _drenar_para_filas(timeout_long_poll=25)
    return None


def aguardar_midia_ou_texto(timeout_s=1800, download_dir=None):
    """Espera a PRÓXIMA mensagem (não-callback): foto, vídeo, documento ou texto.
    Mesma lógica de fila-primeiro de aguardar_callback. Retorna None no timeout."""
    limite = time.time() + timeout_s
    while time.time() < limite:
        if _fila_mensagens:
            return _fila_mensagens.pop(0)
        _drenar_para_filas(timeout_long_poll=25)
    return None


def _aguardar_callback_ou_midia(timeout_s=1800):
    """Espera OU um clique de botão OU uma mídia/texto direto — o que chegar primeiro.
    É isso que permite responder um clipe SEM precisar clicar Recusar→Enviar mídia:
    manda a foto/link direto que já vale como substituição. Retorna (None, None) no
    timeout, ('callback', data) ou ('mensagem', dict)."""
    limite = time.time() + timeout_s
    while time.time() < limite:
        if _fila_callbacks:
            return ('callback', _fila_callbacks.pop(0))
        if _fila_mensagens:
            return ('mensagem', _fila_mensagens.pop(0))
        _drenar_para_filas(timeout_long_poll=25)
    return (None, None)


_contador_downloads = itertools.count(1)


def _baixar_arquivo_telegram(file_id, download_dir, extensao):
    info = _chamar('getFile', file_id=file_id)
    file_path = info['file_path']
    url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    # BUGFIX (fotos enviadas pelo Telegram apareciam repetidas e fora de ordem):
    # o nome era montado com file_id[:16], mas os file_id do Telegram compartilham
    # um prefixo longo (tipo + datacenter + chat). Duas fotos diferentes do mesmo
    # chat geravam EXATAMENTE o mesmo nome, então cada download sobrescrevia o
    # anterior — e na hora de renderizar todos os slots liam o mesmo arquivo,
    # mostrando a mesma mídia várias vezes. Agora o nome usa o hash do file_id
    # INTEIRO (sem truncar) mais um contador, garantindo nome único por download.
    assinatura = hashlib.sha1(file_id.encode('utf-8')).hexdigest()[:20]
    seq = next(_contador_downloads)
    destino = os.path.join(download_dir, f"tg_{seq:03d}_{assinatura}{extensao}")
    resp = com_watchdog(requests.get, url, timeout=60,
                         timeout_total=90, label="download de arquivo do Telegram")
    if resp is None or not resp.ok:
        raise RuntimeError(f"Falha ao baixar arquivo do Telegram (file_id={file_id})")
    with open(destino, 'wb') as f:
        f.write(resp.content)
    if extensao.lower() in ('.jpg', '.jpeg', '.png'):
        _normalizar_imagem(destino)
    return destino


def _normalizar_imagem(caminho, largura_max=1920):
    """
    BUGFIX (renderização travando em ~4%, sempre no mesmo frame): foto mandada direto
    do celular pelo Telegram costuma vir em 3000-4000px+ de largura — bem maior que
    qualquer imagem que o pipeline automático já lida (Pexels/Wikimedia já vêm em
    resolução moderada). O efeito de zoom (_clip_de_imagem_com_zoom, em
    generate_video.py) reprocessa a imagem A CADA FRAME pra animar o zoom — com uma
    foto de celular gigante sem redimensionar antes, cada frame fica MUITO mais caro
    de calcular, e como o zoom dura o clipe inteiro, a exportação não trava de vez,
    só fica absurdamente lenta (segundos por frame em vez de frames por segundo) —
    o que parece travado mas é só um vídeo de ~30-60s de duração real do clipe
    levando dezenas de minutos pra renderizar.

    Redimensiona pra no máximo 'largura_max' no lado maior (de sobra pra qualquer
    resolução de saída do vídeo, sem carregar peso à toa) e corrige a rotação EXIF
    (foto de celular quase sempre tem isso, e sem corrigir alguns leitores mostram a
    imagem de lado). Roda uma vez, no download — nunca durante a renderização.
    """
    try:
        img = Image.open(caminho)
        img = ImageOps.exif_transpose(img)  # corrige rotação de foto de celular
        if img.mode not in ('RGB',):
            img = img.convert('RGB')
        if max(img.size) > largura_max:
            escala = largura_max / max(img.size)
            novo_tamanho = (max(1, int(img.width * escala)), max(1, int(img.height * escala)))
            img = img.resize(novo_tamanho, Image.LANCZOS)
            print(f"    🖼️ Imagem redimensionada pra {novo_tamanho[0]}x{novo_tamanho[1]} "
                  f"antes de entrar no pipeline (estava maior que {largura_max}px)")
        img.save(caminho, quality=90)
    except Exception as e:
        print(f"    ⚠️ Falha ao normalizar imagem '{caminho}' ({e}) — usando como veio, "
              f"pode deixar a renderização mais lenta se for muito grande")


# ============================================================
# Substituição de mídia via link do Pexels
# ============================================================

def _extrair_id_pexels(url):
    """Extrai o ID numérico do fim de uma URL de vídeo/foto do Pexels
    (ex: .../video/aerial-city-1234567/ -> '1234567')."""
    m = re.search(r'-(\d+)/?(?:$|[?#])', url.strip())
    return m.group(1) if m else None


def _baixar_pexels_por_id(url, download_dir=None, largura_alvo=1920):
    """Aceita link de /video/ ou /photo/ do Pexels e baixa a variante daquele item
    específico mais próxima de 'largura_alvo' (não uma busca — o ID exato que a URL
    aponta). Retorna o caminho local ou None se não conseguir resolver (URL sem ID,
    sem chave de API, ou erro de rede — sempre logado, nunca deixa a revisão travada
    sem explicação).

    BUGFIX (vídeo suspeito de travar a renderização em ~4%): antes pegava sempre a
    MAIOR resolução disponível (podia vir 4K de um vídeo Pexels) — bem mais pesado pra
    decodificar/redimensionar que o necessário, já que o vídeo final não passa da
    resolução configurada mesmo. O resto do pipeline (baixar_clipes_pexels,
    _escolher_arquivo_video) já escolhe a variante mais PRÓXIMA da largura alvo, não a
    maior — agora aqui faz o mesmo, por consistência e performance."""
    download_dir = download_dir or os.path.join('assets', 'telegram_review')
    os.makedirs(download_dir, exist_ok=True)

    pexels_id = _extrair_id_pexels(url)
    if not pexels_id:
        print(f"    ⚠️ Não achei um ID de item do Pexels nesse link: '{url}'")
        return None
    if not PEXELS_API_KEY:
        print("    ⚠️ PEXELS_API_KEY não configurada — não dá pra baixar por link do Pexels")
        return None

    headers = {"Authorization": PEXELS_API_KEY}
    eh_foto = '/photo/' in url or '/foto/' in url
    try:
        if eh_foto:
            resp = requests.get(f"https://api.pexels.com/v1/photos/{pexels_id}",
                                 headers=headers, timeout=20)
            resp.raise_for_status()
            src = resp.json().get('src', {})
            link = src.get('large2x') or src.get('original')
            destino = os.path.join(download_dir, f"pexels_photo_{pexels_id}.jpg")
        else:
            resp = requests.get(f"https://api.pexels.com/videos/videos/{pexels_id}",
                                 headers=headers, timeout=20)
            resp.raise_for_status()
            arquivos = [vf for vf in resp.json().get('video_files', []) if vf.get('link') and vf.get('width')]
            link = min(arquivos, key=lambda vf: abs(vf['width'] - largura_alvo))['link'] if arquivos else None
            destino = os.path.join(download_dir, f"pexels_video_{pexels_id}.mp4")

        if not link:
            print(f"    ⚠️ Item {pexels_id} do Pexels não tem arquivo baixável")
            return None

        conteudo = com_watchdog(requests.get, link, timeout=60,
                                 timeout_total=90, label=f"download Pexels {pexels_id}")
        if conteudo is None or not conteudo.ok:
            return None
        with open(destino, 'wb') as f:
            f.write(conteudo.content)
        if eh_foto:
            _normalizar_imagem(destino, largura_max=largura_alvo)
        return destino
    except Exception as e:
        print(f"    ⚠️ Falha ao baixar item {pexels_id} do Pexels ({e})")
        return None


# ============================================================
# Trecho de roteiro correspondente a uma janela de tempo
# ============================================================

def _trecho_do_roteiro(texto_segmento, palavras_tempo, inicio, fim):
    """Reconstrói o trecho do roteiro ORIGINAL (nunca o texto que o Whisper
    reconheceu — só o TEMPO dele é usado) narrado dentro de [inicio, fim), mesmo
    pareamento posicional usado em mapear_tempos_para_blocos/gerar_clips_legenda."""
    palavras = texto_segmento.split()
    n = min(len(palavras), len(palavras_tempo))
    indices = [i for i in range(n)
               if palavras_tempo[i]['fim'] > inicio and palavras_tempo[i]['inicio'] < fim]
    if not indices:
        return "(sobra de tempo sem palavra mapeada — provavelmente o fim de um bloco)"
    return " ".join(palavras[indices[0]:indices[-1] + 1])


# ============================================================
# Revisão de mídia, clipe a clipe
# ============================================================

def revisar_midia_pipeline(lista_clipes, texto_segmento, palavras_tempo, nome_segmento,
                            largura_alvo=1920):
    """
    Chamada de dentro de renderizar_segmento_webdoc, DEPOIS de baixar_clipes_por_bloco
    e ANTES de montar o vídeo (_montar_clips_pexels) — lista_clipes ainda está em
    formato bruto [{'path','inicio','duracao',...}], tempo relativo ao início da
    narração do segmento (mesmo referencial de texto_segmento/palavras_tempo).

    Cada clipe pode ser respondido de DOIS jeitos, sem precisar escolher um só:
      - clicando ✅ Aprovar / ❌ Recusar
      - mandando a substituição DIRETO (link do Pexels ou foto/vídeo do aparelho),
        sem precisar clicar em nada antes — vale como "recusar + já aqui está a mídia"
    Isso deixa mandar tudo em sequência rápida (como a maioria das pessoas naturalmente
    faz) sem precisar esperar o bot reperguntar a cada clipe.

    largura_alvo: passado pra _baixar_pexels_por_id — garante que um link de Pexels
    baixe a variante de resolução mais próxima do vídeo final (não a maior disponível,
    que pode ser 4K e pesar bem mais na hora de renderizar sem ganho nenhum de
    qualidade perceptível no resultado final).

    Devolve a MESMA lista, com 'path' trocado nos clipes que foram substituídos.
    Levanta WorkflowCanceladoPeloUsuario se o usuário cancelar.
    """
    if not ATIVA_TELEGRAM or not lista_clipes:
        return lista_clipes

    timeout_min = _timeout_min('telegram_review', 30)
    # BUGFIX: nunca herdar uma mensagem/callback que sobrou sem uso de um segmento ou
    # clipe anterior — impede vazamento tipo "mídia mandada pra um clipe da introdução
    # aparecendo no capítulo 1".
    limpar_filas_pendentes()

    print(f"  📲 Revisão de mídia via Telegram — segmento '{nome_segmento}' "
          f"({len(lista_clipes)} clipe(s), timeout {timeout_min} min/resposta)...")

    enviar_texto(f"🎬 Revisão de mídia — segmento \"{nome_segmento}\"\n"
                 f"{len(lista_clipes)} clipe(s) pra aprovar, um de cada vez. Pode "
                 f"aprovar/recusar pelos botões OU já mandar a substituição direto "
                 f"(link do Pexels ou foto/vídeo do aparelho) que eu entendo.")

    # Uma única chamada de tradução por segmento (não uma por clipe): trecho NO + PT-BR +
    # sugestão de que mídia procurar, tudo em PT-BR pra quem opera não precisar ler o idioma.
    trechos = [_trecho_do_roteiro(texto_segmento, palavras_tempo,
                                  c['inicio'], c['inicio'] + c['duracao']) for c in lista_clipes]
    traduziveis = [i for i, t in enumerate(trechos) if not t.startswith('(sobra de tempo')]
    traducoes = [{'pt': None, 'sugestao': None, 'busca_en': None} for _ in trechos]
    for i, t in zip(traduziveis, _traduzir([trechos[i] for i in traduziveis], com_sugestao=True)):
        traducoes[i] = t

    for i, clipe in enumerate(lista_clipes):
        trecho = trechos[i]
        dur_exibida = clipe.get('duracao_exibida', clipe['duracao'])
        legenda = (f"Clipe {i + 1}/{len(lista_clipes)} — {dur_exibida:.1f}s "
                   f"(fonte: {clipe.get('fonte', 'pexels')})\n\n\"{trecho[:330]}\"")
        if traducoes[i]['pt']:
            legenda += f"\n\n🇧🇷 \"{traducoes[i]['pt'][:330]}\""
        if traducoes[i]['sugestao']:
            legenda += f"\n\n💡 Mídia sugerida: {traducoes[i]['sugestao'][:160]}"
        termo_busca = traducoes[i].get('busca_en') or traducoes[i]['sugestao']
        if termo_busca and traducoes[i].get('busca_en'):
            legenda += f"\n🔎 Busca: {termo_busca[:80]}"

        botoes_clipe = [('✅ Aprovar', 'aprovar'), ('❌ Recusar', 'recusar')]
        url_pexels = _url_busca_pexels(termo_busca)
        if url_pexels:
            botoes_clipe = [botoes_clipe, [('🔎 Pesquisar no Pexels', 'url:' + url_pexels)]]
        enviar_midia(clipe['path'], legenda, botoes=botoes_clipe)

        tipo, valor = _aguardar_callback_ou_midia(timeout_s=timeout_min * 60)

        if tipo is None:
            print(f"    ⏱️ Sem resposta em {timeout_min} min pro clipe {i + 1} — "
                  f"aprovando automaticamente pra não travar o pipeline")
            continue

        if tipo == 'callback' and valor == 'aprovar':
            continue

        novo_caminho = None

        if tipo == 'mensagem':
            # usuário já mandou a substituição direto, sem clicar em nada
            novo_caminho = (_baixar_pexels_por_id(valor['texto'], largura_alvo=largura_alvo)
                             if valor['tipo'] == 'texto' else valor['caminho'])
            if not novo_caminho:
                enviar_texto(f"⚠️ Não consegui usar essa mídia pro clipe {i + 1} — "
                             f"mantendo o clipe original.")

        else:  # tipo == 'callback' e valor == 'recusar' → fluxo explícito de botões
            while novo_caminho is None:
                enviar_texto("O que fazer com esse clipe?",
                             botoes=[('🚫 Cancelar workflow', 'cancelar'), ('📤 Enviar mídia', 'enviar')])
                escolha = aguardar_callback(timeout_s=timeout_min * 60)

                if escolha is None or escolha == 'cancelar':
                    enviar_texto("🚫 Workflow cancelado. Nenhum vídeo será publicado.")
                    raise WorkflowCanceladoPeloUsuario(
                        f"Cancelado pelo usuário no clipe {i + 1} do segmento '{nome_segmento}'")

                enviar_texto("Manda o link do Pexels (pexels.com/video/... ou /photo/...) "
                             "ou envie a foto/vídeo direto daqui.")
                recebido = aguardar_midia_ou_texto(timeout_s=timeout_min * 60)

                if recebido is None:
                    enviar_texto(f"⏱️ Sem resposta em {timeout_min} min — mantendo o clipe original.")
                    break

                novo_caminho = (_baixar_pexels_por_id(recebido['texto'], largura_alvo=largura_alvo)
                                 if recebido['tipo'] == 'texto' else recebido['caminho'])
                if not novo_caminho:
                    enviar_texto("⚠️ Não consegui usar essa mídia — manda de novo, ou cancela.")

        if novo_caminho:
            clipe['path'] = novo_caminho
            clipe['fonte'] = 'telegram_manual'
            enviar_texto(f"✅ Clipe {i + 1} substituído, seguindo pro próximo.")

    enviar_texto(f"✅ Revisão do segmento \"{nome_segmento}\" concluída.")
    return lista_clipes


# ============================================================
# Seleção de tema no início do workflow
# ============================================================

# ============================================================
# Áudio customizado (usuário grava/manda a própria narração do segmento)
# ============================================================

def _normalizar_audio_para_segmento(caminho_origem, destino_mp3):
    """
    Converte QUALQUER formato de áudio que o Telegram mande (nota de voz é sempre
    .ogg/Opus; arquivo pode vir em .mp3/.m4a/.wav/etc.) pro mp3 no caminho exato
    que renderizar_segmento_webdoc já espera (audio_path_seg). Depois disso, o
    resto do pipeline (Whisper, AudioFileClip, SFX, música de fundo...) trata esse
    áudio exatamente como trataria uma narração gerada por TTS — nenhum outro
    ponto do código precisa saber que a origem foi manual.
    """
    cmd = ['ffmpeg', '-y', '-i', caminho_origem, '-vn', '-acodec', 'libmp3lame',
           '-q:a', '2', destino_mp3]
    resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if resultado.returncode != 0:
        raise RuntimeError(f"Falha ao converter áudio enviado pra mp3: {resultado.stderr[-500:]}")
    return destino_mp3


def perguntar_audio_customizado_telegram(nome_segmento, timeout_min=None, texto_segmento=None):
    """
    Chamada ANTES de gerar a narração (Fish Audio) de um segmento (introdução,
    capítulo N ou desfecho): pergunta se você quer mandar seu PRÓPRIO áudio pra
    esse trecho, em vez do pipeline gerar por TTS. Evita gastar uma chamada de
    TTS à toa quando você já sabe que vai substituir.

    Pode responder de dois jeitos, igual às outras revisões:
      - clicando um dos botões
      - já mandando o áudio (nota de voz ou arquivo) DIRETO, sem clicar em nada

    O áudio mandado é usado como está — SEM inserir pausas artificiais entre
    frases (essa lógica é só do gerador TTS por trechos, que nem roda nesse
    caminho); o pipeline só aplica as margens de silêncio antes/depois do
    segmento, exatamente como já faz com a narração gerada automaticamente.

    texto_segmento (opcional): o texto que deve ser narrado. Quando vem, é mandado ANTES
    da pergunta, limpo e fácil de copiar pro seu gerador de voz, seguido da tradução em
    PT-BR (se houver tradutor configurado) — necessário quando o canal narra num idioma
    que você não lê.

    config.json -> 'audio_customizado_telegram': {"obrigatorio": true} remove o botão
    "Gerar automático": o áudio passa a ser a ÚNICA opção, e sem resposta no timeout o
    workflow é CANCELADO (WorkflowCanceladoPeloUsuario) em vez de publicar um vídeo com
    voz de fallback num idioma onde ela não foi pensada.

    Devolve o CAMINHO do mp3 já convertido, ou None se você decidir não mandar
    (ou não responder a tempo — nesse caso segue com TTS normal, não trava o
    pipeline esperando; exceto no modo obrigatório, descrito acima).
    """
    if not ATIVA_TELEGRAM:
        return None
    cfg_audio = _config().get('audio_customizado_telegram', {})
    obrigatorio = bool(cfg_audio.get('obrigatorio', False))
    if timeout_min is None:
        timeout_min = _timeout_min('audio_customizado_telegram', 15)

    limpar_filas_pendentes()

    if texto_segmento:
        enviar_texto_com_traducao(
            texto_segmento,
            cabecalho_original=f"🎙️ Texto do segmento \"{nome_segmento}\" — copie pro seu gerador de voz:")

    if obrigatorio:
        enviar_texto(
            f"🎙️ Segmento \"{nome_segmento}\": mande o ÁUDIO desse texto (nota de voz ou "
            f"arquivo de áudio). O vídeo só segue depois que o áudio chegar.",
            botoes=[('🚫 Cancelar workflow', 'cancelar')]
        )
    else:
        enviar_texto(
            f"🎙️ Segmento \"{nome_segmento}\": quer mandar seu PRÓPRIO áudio pra esse "
            f"trecho, em vez de eu gerar a narração? Pode ser nota de voz ou arquivo "
            f"de áudio.",
            botoes=[('🎙️ Vou mandar', 'vou_mandar'), ('🤖 Gerar automático', 'automatico')]
        )

    while True:
        tipo, valor = _aguardar_callback_ou_midia(timeout_s=timeout_min * 60)

        if tipo is None:
            if obrigatorio:
                enviar_texto(f"⏱️ Sem áudio em {timeout_min} min — workflow cancelado, nada foi publicado.")
                raise WorkflowCanceladoPeloUsuario(
                    f"Sem áudio recebido pro segmento '{nome_segmento}' em {timeout_min} min")
            print(f"    ⏱️ Sem resposta em {timeout_min} min — gerando narração automática")
            return None

        if tipo == 'callback' and valor == 'cancelar':
            enviar_texto("🚫 Workflow cancelado. Nenhum vídeo será publicado.")
            raise WorkflowCanceladoPeloUsuario(
                f"Cancelado pelo usuário no áudio do segmento '{nome_segmento}'")

        if tipo == 'callback' and valor == 'automatico' and not obrigatorio:
            print("    🤖 Optou por narração automática pra este segmento")
            return None

        if tipo == 'callback' and valor == 'vou_mandar':
            enviar_texto("Manda o áudio (nota de voz ou arquivo).")
            continue

        if tipo == 'mensagem' and valor['tipo'] == 'audio':
            pasta = _pasta_downloads_padrao()
            destino_mp3 = os.path.join(pasta, f"audio_custom_{nome_segmento}.mp3")
            try:
                _normalizar_audio_para_segmento(valor['caminho'], destino_mp3)
            except Exception as e:
                enviar_texto(f"⚠️ Não consegui processar esse áudio ({e}). Manda de novo"
                             + ("." if obrigatorio else ", ou aperta 🤖 Gerar automático."))
                continue
            print(f"    🎙️ Áudio próprio recebido pra '{nome_segmento}'")
            enviar_texto(f"✅ Áudio de \"{nome_segmento}\" recebido.")
            return destino_mp3

        # texto solto, foto, vídeo ou documento não servem de áudio
        enviar_texto("⚠️ Preciso de um ÁUDIO (nota de voz ou arquivo de áudio). Manda de "
                     "novo" + ("." if obrigatorio else ", ou aperta 🤖 Gerar automático pra eu gerar a narração."))


# ============================================================
# Escolha manual de palavras de destaque
# ============================================================

def escolher_destaques_telegram(texto_segmento, nome_segmento, timeout_min=None):
    """
    Manda o texto INTEIRO do segmento e pergunta quais expressões destacar, em vez de
    deixar o Gemini escolher sozinho (escolher_palavras_destaque, em producao_visual.py).
    Resposta esperada: uma expressão por linha, EXATAMENTE como aparece no texto
    mandado (mapear_destaques_manuais_para_blocos, em producao_visual.py, faz a
    checagem literal depois). Botão "Automático" ou timeout → retorna None, e quem
    chama cai pra escolher_palavras_destaque() de sempre.
    """
    if not ATIVA_TELEGRAM:
        return None
    if timeout_min is None:
        timeout_min = _timeout_min('destaques_telegram', 15)

    enviar_texto_com_traducao(
        texto_segmento,
        cabecalho_original=f"✨ Segmento \"{nome_segmento}\" — texto original (copie as expressões daqui):")
    enviar_texto(
        "Quais palavras/expressões quer destacar? Responda com uma expressão POR LINHA, "
        "EXATAMENTE como está escrita no texto acima (1 a 3 palavras cada, é assim que "
        "vão aparecer na tela) — copie e cole do texto original, não da tradução. "
        "Ou aperta o botão pra deixar o Gemini escolher automático.",
        botoes=[('🤖 Automático', 'automatico')]
    )

    limite = time.time() + timeout_min * 60
    while time.time() < limite:
        for upd in _proximos_updates(timeout_long_poll=25):
            cq = upd.get('callback_query')
            if cq and str(cq.get('message', {}).get('chat', {}).get('id')) == str(TELEGRAM_CHAT_ID):
                try:
                    _chamar('answerCallbackQuery', callback_query_id=cq['id'])
                except Exception:
                    pass
                if cq.get('data') == 'automatico':
                    print("  🤖 Usuário escolheu destaque automático")
                    return None
            msg = upd.get('message')
            if msg and str(msg.get('chat', {}).get('id')) == str(TELEGRAM_CHAT_ID) and msg.get('text'):
                frases = [l.strip() for l in msg['text'].splitlines() if l.strip()]
                if frases:
                    enviar_texto(f"✨ {len(frases)} expressão(ões) marcada(s) pra destaque.")
                    print(f"  ✨ Destaques manuais recebidos via Telegram: {frases}")
                    return frases

    print(f"  ⏱️ Sem resposta de destaques em {timeout_min} min — automático")
    return None


def _ajustar_thumbnail_16x9(caminho, largura=1280, altura=720):
    """Corta (cover-crop, sem distorcer) e redimensiona pro tamanho padrão de
    thumbnail do YouTube — uma foto mandada manualmente do celular raramente já vem
    exatamente nessa proporção."""
    try:
        img = Image.open(caminho)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.fit(img, (largura, altura), Image.LANCZOS)
        img.save(caminho, quality=92)
    except Exception as e:
        print(f"    ⚠️ Falha ao ajustar thumbnail pro formato 16:9 ({e}) — usando como veio")
    return caminho


def revisar_thumbnail_telegram(thumbnail_path, timeout_min=None, legenda_extra=""):
    """
    Chamada logo depois de gerar_thumbnail() e ANTES de fazer_upload_youtube(): manda
    a thumbnail gerada com [✅ Aprovar]/[📤 Enviar outra]. Pode responder de dois
    jeitos, igual à revisão de mídia:
      - clicando um dos botões
      - mandando a foto substituta DIRETO, sem precisar clicar em nada antes
    A substituta é ajustada pro formato 1280x720 (padrão de thumbnail do YouTube)
    antes de virar a definitiva. Sem resposta dentro do timeout → aprova
    automaticamente a gerada (não trava a publicação esperando pra sempre).

    Se ATIVA_TELEGRAM for False, ou 'thumbnail_path' vier None (gerar_thumbnail()
    já falhou sozinha), não faz nada — devolve o que recebeu, no mesmo espírito de
    "publica normalmente, só sem thumbnail customizada" que gerar_thumbnail() usa.
    """
    if not ATIVA_TELEGRAM or not thumbnail_path:
        return thumbnail_path
    if timeout_min is None:
        timeout_min = _timeout_min('thumbnail_telegram', 15)

    limpar_filas_pendentes()
    print(f"  📲 Aprovação de thumbnail via Telegram (timeout {timeout_min} min)...")

    caminho_atual = thumbnail_path
    sufixo = f"\n\n{legenda_extra}" if legenda_extra else ""
    enviar_midia(caminho_atual, "🖼️ Thumbnail gerada — aprovar ou mandar outra?" + sufixo,
                 botoes=[('✅ Aprovar', 'aprovar'), ('📤 Enviar outra', 'substituir')])

    while True:
        tipo, valor = _aguardar_callback_ou_midia(timeout_s=timeout_min * 60)

        if tipo is None:
            print(f"    ⏱️ Sem resposta em {timeout_min} min — aprovando a thumbnail gerada")
            return caminho_atual

        if tipo == 'callback' and valor == 'aprovar':
            print("    ✅ Thumbnail aprovada")
            return caminho_atual

        if tipo == 'callback' and valor == 'substituir':
            enviar_texto("Manda a foto que vai substituir a thumbnail.")
            continue

        if tipo == 'mensagem' and valor['tipo'] == 'foto':
            caminho_atual = _ajustar_thumbnail_16x9(valor['caminho'])
            print("    📤 Thumbnail substituída pela recebida no Telegram")
            enviar_midia(caminho_atual, "🖼️ Nova thumbnail — aprovar ou mandar outra?" + sufixo,
                         botoes=[('✅ Aprovar', 'aprovar'), ('📤 Enviar outra', 'substituir')])
            continue

        # texto solto, vídeo ou documento não servem de thumbnail
        enviar_texto("⚠️ Preciso de uma FOTO (jpg/png) pra usar como thumbnail. Manda de novo, "
                     "ou aperta ✅ Aprovar pra manter a atual.")


def escolher_formato_telegram(timeout_min=None):
    """
    Primeira pergunta da interação: qual a ESTRUTURA do vídeo.
      • 📽️ Webdoc: introdução → capítulos → desfecho, com card preto entre capítulos
      • 🔢 Lista ("10 motos que...", "7 comidas que..."): introdução → itens numerados →
        desfecho, com um card de número + nome do item sobre a mídia (sem tela preta)

    Pra lista, pergunta também: quantos itens, a ordem (regressiva N→1 ou crescente 1→N) e
    quem escolhe os NOMES dos itens (o pipeline/Gemini, ou você digitando um por linha).

    Devolve {'formato': 'webdoc'} ou
            {'formato': 'lista', 'num_itens': N, 'ordem': 'regressiva'|'crescente',
             'itens': [nomes] | None}
    Timeout/sem Telegram → o formato padrão do config ('formato_video_telegram' →
    'formato_padrao', normalmente 'webdoc'), então o workflow nunca trava esperando.
    """
    cfg = _config().get('formato_video_telegram', {})
    padrao = {'formato': cfg.get('formato_padrao', 'webdoc')}
    if not ATIVA_TELEGRAM or not cfg.get('ativo', False):
        return padrao
    if timeout_min is None:
        timeout_min = _timeout_min('formato_video_telegram', 10)
    espera = timeout_min * 60

    def _perguntar_botoes(texto, botoes):
        enviar_texto(texto, botoes=botoes)
        while True:
            tipo, valor = _aguardar_callback_ou_midia(timeout_s=espera)
            if tipo is None:
                return None
            if tipo == 'callback':
                return valor

    limpar_filas_pendentes()
    escolha = _perguntar_botoes("🎬 Qual a estrutura do vídeo?",
                                [('📽️ Webdoc (capítulos)', 'webdoc'), ('🔢 Lista (itens numerados)', 'lista')])
    if escolha != 'lista':
        enviar_texto("👍 Formato: webdoc (introdução → capítulos → desfecho).")
        return {'formato': 'webdoc'} if escolha == 'webdoc' else padrao

    # ── quantos itens ──
    enviar_texto("🔢 Quantos itens na lista? Toque num número ou digite (de 3 a 30).",
                 botoes=[('5', 'n:5'), ('7', 'n:7'), ('10', 'n:10')])
    num_itens = None
    while num_itens is None:
        tipo, valor = _aguardar_callback_ou_midia(timeout_s=espera)
        if tipo is None:
            num_itens = 7
        elif tipo == 'callback' and valor.startswith('n:'):
            num_itens = int(valor[2:])
        elif tipo == 'mensagem' and valor['tipo'] == 'texto' and valor['texto'].isdigit() \
                and 3 <= int(valor['texto']) <= 30:
            num_itens = int(valor['texto'])
        elif tipo == 'mensagem':
            enviar_texto("⚠️ Manda só um número de 3 a 30, ou toque num botão.")

    # ── ordem ──
    escolha = _perguntar_botoes(
        f"↕️ Ordem dos {num_itens} itens?",
        [(f'⬇️ Regressiva ({num_itens}→1)', 'regressiva'), (f'⬆️ Crescente (1→{num_itens})', 'crescente')])
    ordem = escolha if escolha in ('regressiva', 'crescente') else 'regressiva'

    # ── nomes dos itens ──
    escolha = _perguntar_botoes("📝 Quem escolhe os itens da lista?",
                                [('🤖 Você escolhe', 'pipeline'), ('✍️ Eu escolho', 'eu')])
    itens = None
    if escolha == 'eu':
        enviar_texto(
            "✍️ Mande os nomes dos itens, UM POR LINHA, na ORDEM em que vão APARECER no vídeo "
            f"(a primeira linha aparece primeiro" +
            (f", e vai ganhar o número {num_itens})." if ordem == 'regressiva' else ", e vai ganhar o número 1).") +
            "\nSe mandar uma quantidade diferente de linhas, uso a quantidade que você mandar.",
            botoes=[('🤖 Deixa que eu escolho', 'pipeline')])
        while True:
            tipo, valor = _aguardar_callback_ou_midia(timeout_s=espera)
            if tipo is None or (tipo == 'callback' and valor == 'pipeline'):
                break
            if tipo == 'mensagem' and valor['tipo'] == 'texto':
                linhas = [re.sub(r'^\s*(?:\d+\s*[.):\-]\s*|[-•*]\s*)', '', l).strip()
                          for l in valor['texto'].splitlines()]
                linhas = [l for l in linhas if l]
                if 3 <= len(linhas) <= 30:
                    itens, num_itens = linhas, len(linhas)
                    break
                enviar_texto("⚠️ Preciso de 3 a 30 itens, um por linha.")
            elif tipo == 'mensagem':
                enviar_texto("⚠️ Preciso dos nomes em texto, um por linha.")

    resumo = (f"👍 Lista de {num_itens} itens, ordem {ordem}, " +
              ("itens escolhidos por você:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(itens))
               if itens else "itens escolhidos pelo pipeline."))
    enviar_texto(resumo)
    return {'formato': 'lista', 'num_itens': num_itens, 'ordem': ordem, 'itens': itens}


def escolher_tema_telegram(timeout_min=None):
    """
    Pergunta o tema/direcionamento do próximo vídeo, com botão "Nada a sugerir". Texto
    de resposta vira o tema (não precisa ser um título pronto — pode ser só o
    direcionamento, ex: "Braskem, o estrago que fez em Maceió com a extração de
    salgema"). Botão OU timeout → retorna None, e quem chama cai pra escolha automática
    (escolher_tema_reflexao(), dentro de generate_video.py).
    """
    if not ATIVA_TELEGRAM:
        return None
    if timeout_min is None:
        timeout_min = _timeout_min('selecao_tema_telegram', 10)

    enviar_texto(
        "🎯 Qual o tema do próximo vídeo? Pode mandar só o direcionamento, não precisa "
        "ser o título pronto (ex: \"a morosidade da transposição do Rio São Francisco\"). "
        "Ou aperta o botão se quiser que eu escolha.",
        botoes=[('🤖 Nada a sugerir', 'nada_a_sugerir')]
    )

    limite = time.time() + timeout_min * 60
    while time.time() < limite:
        for upd in _proximos_updates(timeout_long_poll=25):
            cq = upd.get('callback_query')
            if cq and str(cq.get('message', {}).get('chat', {}).get('id')) == str(TELEGRAM_CHAT_ID):
                try:
                    _chamar('answerCallbackQuery', callback_query_id=cq['id'])
                except Exception:
                    pass
                if cq.get('data') == 'nada_a_sugerir':
                    print("  🤖 Usuário escolheu 'Nada a sugerir' — tema automático")
                    return None
            msg = upd.get('message')
            if msg and str(msg.get('chat', {}).get('id')) == str(TELEGRAM_CHAT_ID) and msg.get('text'):
                tema = msg['text'].strip()
                enviar_texto(f"👍 Tema recebido: \"{tema}\". Gerando o roteiro...")
                print(f"  🎯 Tema recebido via Telegram: \"{tema}\"")
                return tema

    print(f"  ⏱️ Sem resposta de tema em {timeout_min} min — escolhendo automaticamente")
    return None


# ============================================================
# Publicação: agora ou agendada
# ============================================================

def _parse_data_hora(texto, tz, agora=None):
    """
    Interpreta a data/hora digitada no Telegram, no fuso 'tz' (zoneinfo). Aceita:
      "25/12 18:30" · "25/12/2026 18:30" · "amanhã 19:00" · "hoje 21h" · "18:30" · "18h"
    Hora sozinha = hoje, ou amanhã se já passou. Devolve datetime com fuso, ou None se
    não entendeu (quem chama pede de novo — nunca adivinha uma data).
    """
    from datetime import datetime, timedelta
    t = texto.strip().lower().replace('às', ' ').replace('as ', ' ')
    t = re.sub(r'\s+', ' ', t)
    agora = agora or datetime.now(tz)

    m_hora = re.search(r'(\d{1,2})\s*(?:[:h]\s*(\d{2})?)\s*$', t)
    if not m_hora:
        return None
    h, mi = int(m_hora.group(1)), int(m_hora.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    resto = t[:m_hora.start()].strip()

    base = None
    if resto in ('', 'hoje'):
        base = agora
    elif resto in ('amanhã', 'amanha'):
        base = agora + timedelta(days=1)
    else:
        m = re.fullmatch(r'(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?', resto)
        if not m:
            return None
        dia, mes = int(m.group(1)), int(m.group(2))
        ano = int(m.group(3)) if m.group(3) else agora.year
        if ano < 100:
            ano += 2000
        try:
            base = datetime(ano, mes, dia, tzinfo=tz)
            if not m.group(3) and base.date() < agora.date():
                base = datetime(ano + 1, mes, dia, tzinfo=tz)  # "03/01" dito em dezembro = ano que vem
        except ValueError:
            return None
    try:
        alvo = datetime(base.year, base.month, base.day, h, mi, tzinfo=tz)
    except ValueError:
        return None
    if resto == '' and alvo <= agora:
        alvo += timedelta(days=1)
    return alvo


def escolher_publicacao_telegram(timeout_min=None):
    """
    Chamada depois da aprovação da thumbnail e ANTES do upload: pergunta se o vídeo vai ao
    ar agora ou agendado. Devolve um datetime (UTC) pra agendar, ou None pra publicar já.
    O agendamento usa o recurso nativo do YouTube (vídeo sobe privado e vira público sozinho
    na hora marcada) — o workflow NÃO fica esperando até lá.

    config.json -> 'agendamento_telegram': {
        "ativo": true, "fuso_horario": "America/Sao_Paulo",     # fuso em que VOCÊ digita
        "fuso_exibicao_extra": "Europe/Oslo",                     # opcional: mostra também esse horário
        "se_sem_resposta": "agora",                               # ou "cancelar"
        "timeout_resposta_min": 60 }
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    if not ATIVA_TELEGRAM:
        return None
    cfg = _config().get('agendamento_telegram', {})
    if not cfg.get('ativo', False):
        return None
    if timeout_min is None:
        timeout_min = _timeout_min('agendamento_telegram', 60)
    tz = ZoneInfo(cfg.get('fuso_horario', 'America/Sao_Paulo'))
    tz_extra = ZoneInfo(cfg['fuso_exibicao_extra']) if cfg.get('fuso_exibicao_extra') else None
    sem_resposta = cfg.get('se_sem_resposta', 'agora')

    def _fmt(dt):
        txt = dt.astimezone(tz).strftime('%d/%m/%Y às %H:%M') + f" ({tz.key})"
        if tz_extra:
            txt += "\n   = " + dt.astimezone(tz_extra).strftime('%d/%m/%Y às %H:%M') + f" ({tz_extra.key})"
        return txt

    instrucao = ("Manda dia e hora, no seu horário (" + tz.key + "). Exemplos:\n"
                 "• 25/12 18:30\n• amanhã 19:00\n• hoje 21h\n• 18:30 (hoje, ou amanhã se já passou)")

    limpar_filas_pendentes()
    enviar_texto("📅 Quando publicar o vídeo?",
                 botoes=[('🚀 Publicar agora', 'agora'), ('📅 Agendar', 'agendar')])

    candidato = None
    while True:
        tipo, valor = _aguardar_callback_ou_midia(timeout_s=timeout_min * 60)

        if tipo is None:
            if sem_resposta == 'cancelar':
                enviar_texto(f"⏱️ Sem resposta em {timeout_min} min — workflow cancelado, nada foi publicado.")
                raise WorkflowCanceladoPeloUsuario("Sem decisão de publicação no Telegram")
            enviar_texto(f"⏱️ Sem resposta em {timeout_min} min — publicando agora.")
            return None

        if tipo == 'callback' and valor == 'agora':
            return None

        if tipo == 'callback' and valor == 'confirmar' and candidato:
            enviar_texto(f"✅ Agendado para {_fmt(candidato)}.")
            return candidato.astimezone(timezone.utc)

        if tipo == 'callback' and valor in ('agendar', 'alterar'):
            candidato = None
            enviar_texto(instrucao)
            continue

        if tipo == 'mensagem' and valor['tipo'] == 'texto':
            alvo = _parse_data_hora(valor['texto'], tz)
            if alvo is None:
                enviar_texto("⚠️ Não entendi essa data/hora. " + instrucao)
                continue
            if alvo < datetime.now(tz) + timedelta(minutes=15):
                enviar_texto("⚠️ Essa hora já passou ou está a menos de 15 min. Manda uma data futura.")
                continue
            candidato = alvo
            enviar_texto(f"Agendar o vídeo para:\n{_fmt(alvo)}?",
                         botoes=[('✅ Confirmar', 'confirmar'), ('✏️ Alterar', 'alterar'),
                                 ('🚀 Agora', 'agora')])
            continue

        enviar_texto("⚠️ Preciso de uma data e hora em texto, ou use os botões.")
