"""
motor.py — análise, cadeia de master e relatório (sem DAW).

Cadeia (nesta ordem; cada etapa só roda se estiver ligada):
   1. passa-altas suave em 25 Hz
   2. rebalanceamento de stems (voz / bateria / baixo, aproximado)
   3. match EQ com a referência (curva de 1/3 de oitava, fase linear)
   4. graves e agudos (shelves; automático pela curva do estilo)
   5. EQ mid/side (centro e laterais separados)
   6. EQ dinâmico (embolado, aspereza, sibilância: só quando passam do normal)
   7. compressão multibanda (3 bandas)
   8. compressor de cola
   9. imagem estéreo (grave em mono, largura)
  10. saturação
  11. soft clip (no modo do limitador "Soft clip")
  12. limitador de true peak, com várias passadas até bater o LUFS alvo

A entrada pode ser UMA mix ou VÁRIAS stems (lista de arquivos): as stems são
somadas com o volume/mudo de cada uma e a soma passa pela mesma cadeia.

processar() faz tudo na memória (para ouvir antes); salvar() grava o WAV
24 bits e o relatório. Também roda pela linha de comando:
    python3 motor.py mix.wav --alvo -14 --teto -1
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

import dsp
import stems
from dsp import lim, LIMITE_EQ_DB

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DESTINOS = {  # chave: (texto na tela, LUFS alvo)
    "streaming": ("Streaming (Spotify, YouTube…): -14 LUFS", -14.0),
    "show": ("Show / playback ao vivo: -10 LUFS", -10.0),
    "referencia": ("Igual a uma música de referência", None),
    "personalizado": ("Personalizado (eu escolho o LUFS)", None),
}

# Energia esperada por banda no master (dB em relação ao total), por estilo.
ESTILOS = {
    "geral": ("Geral (pop / rock)", {"sub": -12, "grave": -6, "presenca": -9, "ar": -14}),
    "mpb": ("MPB / bossa / acústico", {"sub": -16, "grave": -7, "presenca": -10, "ar": -16}),
    "reggae": ("Reggae", {"sub": -9, "grave": -4, "presenca": -12, "ar": -19}),
    "funk": ("Funk / black music", {"sub": -10, "grave": -5, "presenca": -10, "ar": -15}),
    "jazz": ("Jazz", {"sub": -16, "grave": -7, "presenca": -10, "ar": -15}),
    "blues": ("Blues", {"sub": -14, "grave": -6, "presenca": -10, "ar": -16}),
    "show": ("Show ao vivo", {"sub": -11, "grave": -5, "presenca": -10, "ar": -15}),
}

COMPRESSAO = {  # cola: chave: (texto, ratio, redução média desejada nas partes fortes)
    "nenhuma": ("Nenhuma", None, 0.0),
    "leve": ("Leve (cola suave)", 1.8, 1.5),
    "media": ("Média (mais pegada)", 2.5, 3.0),
}

DINAMICO = {"desligado": ("Desligado", 0.0), "leve": ("Leve (até 3 dB)", 3.0), "medio": ("Médio (até 6 dB)", 6.0)}

EQ_AUTO_MAX = 2.0
LUFS_MIN, LUFS_MAX = -20.0, -6.0
TETO_MIN, TETO_MAX = -3.0, -0.1

BANDAS = {"sub": (20, 60), "grave": (60, 250), "embolado": (250, 500),
          "medio": (500, 2000), "presenca": (2000, 6000), "ar": (6000, 20000)}

PADRAO = {
    "destino": "streaming", "alvo_lufs": -14.0, "teto_tp": -1.0, "estilo": "geral", "referencia": None,
    "grave": None, "agudo": None, "match_eq": 60,
    "ms_mid": {"grave": 0.0, "presenca": 0.0, "agudo": 0.0}, "ms_side": {"grave": 0.0, "agudo": 0.0},
    "dinamico": "desligado", "dinamico_faixas": ["aspereza", "sibilancia"],
    "multibanda": "nenhuma", "compressao": "leve",
    "mono_grave": True, "mono_grave_hz": 120, "largura": 100,
    "stems": {"voz": 0.0, "bateria": 0.0, "baixo": 0.0},
    "stems_ganhos": {}, "stems_mudos": [],
    "saturacao": False, "limitador": "transparente",
}


def completar(cfg: dict) -> dict:
    c = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in PADRAO.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


# ---------------------------------------------------------------------------
# Leitura e análise
# ---------------------------------------------------------------------------

def ler(caminho: str) -> tuple[np.ndarray, int]:
    """Lê o arquivo como float64 (amostras, canais). Mantém mono ou estéreo."""
    x, sr = sf.read(caminho, dtype="float64", always_2d=True)
    if x.shape[1] > 2:
        x = x[:, :2]
    if len(x) < sr:
        raise ValueError("o arquivo tem menos de 1 segundo de áudio")
    return x, int(sr)


def caminho_saida(entrada) -> str:
    base = nome_base(entrada)
    saida = base + "_master.wav"
    n = 2
    while os.path.exists(saida):
        saida = f"{base}_master_{n}.wav"
        n += 1
    return saida


def nome_base(entrada) -> str:
    """Caminho (sem extensão) usado para nomear o master."""
    if isinstance(entrada, str):
        return os.path.splitext(entrada)[0]
    pasta = os.path.dirname(entrada[0])
    nomes = [os.path.splitext(os.path.basename(c))[0] for c in entrada]
    prefixo = os.path.commonprefix(nomes).rstrip(" -_.#(")
    return os.path.join(pasta, prefixo if len(prefixo) >= 3 else "Master das stems")


def descricao_entrada(entrada) -> str:
    if isinstance(entrada, str):
        return os.path.basename(entrada)
    pasta = os.path.basename(os.path.dirname(entrada[0])) or "."
    return f"{len(entrada)} stems (pasta {pasta})"


def carregar_stems(caminhos: list, ganhos: dict, mudos: list, avisos: list, progresso=None):
    """Soma as stems (uma por vez, para poupar memória) com o volume e o mudo de
    cada uma. Iguala a taxa de amostragem (pela 1ª stem), completa as mais curtas
    com silêncio no fim e transforma mono em estéreo se houver alguma estéreo.
    Devolve (soma, taxa, lista de (nome, ganho dB ou None se mudo))."""
    from math import gcd
    from scipy import signal as sg
    infos = []
    for c in caminhos:
        try:
            infos.append(sf.info(c))
        except Exception as e:
            raise ValueError(f"não consegui abrir a stem {os.path.basename(c)}: {e}")
    sr = int(infos[0].samplerate)
    canais = 2 if any(i.channels >= 2 for i in infos) else 1
    n = max(int(round(i.frames * sr / i.samplerate)) for i in infos)
    if n < sr:
        raise ValueError("as stems têm menos de 1 segundo de áudio")
    soma = np.zeros((n, canais))
    lista, outras_taxas, clipadas = [], set(), []
    for k, (c, info) in enumerate(zip(caminhos, infos)):
        nome = os.path.basename(c)
        if progresso:
            progresso(k + 1, len(caminhos))
        if c in mudos or nome in mudos:
            lista.append((nome, None))
            continue
        g_db = lim(float(ganhos.get(c, ganhos.get(nome, 0.0)) or 0.0), -LIMITE_EQ_DB, LIMITE_EQ_DB)
        x, srx = sf.read(c, dtype="float64", always_2d=True)
        if amostras_clipadas(x) > 10:
            clipadas.append(nome)
        if srx != sr:
            outras_taxas.add(int(srx))
            g = gcd(int(srx), sr)
            x = sg.resample_poly(x, sr // g, int(srx) // g, axis=0)
        x = x[:n, :2]
        g = 10 ** (g_db / 20)
        if x.shape[1] == 1:
            for ch in range(canais):
                soma[: len(x), ch] += x[:, 0] * g
        else:
            soma[: len(x)] += x * g
        lista.append((nome, g_db))
    if all(g is None for _, g in lista):
        raise ValueError("todas as stems estão no mudo")
    if outras_taxas:
        taxas = ", ".join(str(t) for t in sorted(outras_taxas))
        avisos.append(f"Algumas stems estavam em outra taxa ({taxas} Hz); converti tudo para {sr} Hz.")
    if clipadas:
        avisos.append("Stems que já chegaram clipadas (exporte de novo com menos volume): " + ", ".join(clipadas) + ".")
    pico = float(np.max(dsp.abs_max_canais(soma)))
    if pico > 0.89:  # a soma pode passar de 0 dB; abaixa tudo junto (não muda o som)
        soma *= 0.89 / pico
    return soma, sr, lista


def energia_bandas(x: np.ndarray, sr: int) -> dict:
    f, p = dsp.espectro(x, sr)
    total = float(np.sum(p[(f >= 20) & (f <= 20000)])) + 1e-30
    return {nome: dsp.db_pot(float(np.sum(p[(f >= a) & (f < b)])) / total) for nome, (a, b) in BANDAS.items()}


def amostras_clipadas(x: np.ndarray) -> int:
    """Conta trechos de 3+ amostras seguidas grudadas no 0 dBFS."""
    a = dsp.abs_max_canais(x) >= 0.9995
    if not np.any(a):
        return 0
    return int(np.sum(a[:-2] & a[1:-1] & a[2:]))


def analisar(x: np.ndarray, sr: int) -> dict:
    r = {"segundos": len(x) / sr, "canais": x.shape[1], "taxa": sr,
         "pico_db": dsp.db_amp(float(np.max(dsp.abs_max_canais(x)))), "true_peak_db": dsp.true_peak(x),
         "bandas": energia_bandas(x, sr), "clipes": amostras_clipadas(x), "correlacao": dsp.correlacao(x)}
    r.update(dsp.loudness(x, sr))
    return r


def eq_automatico(bandas: dict, curva: dict) -> tuple[float, float]:
    """Metade da diferença entre a mix e a curva alvo, no máximo ±2 dB."""
    dg = max(bandas["sub"], bandas["grave"]) - max(curva["sub"], curva["grave"])
    da = max(bandas["presenca"], bandas["ar"]) - max(curva["presenca"], curva["ar"])
    return (round(lim(-dg / 2, -EQ_AUTO_MAX, EQ_AUTO_MAX), 1),
            round(lim(-da / 2, -EQ_AUTO_MAX, EQ_AUTO_MAX), 1))


# ---------------------------------------------------------------------------
# Processar (na memória) e salvar
# ---------------------------------------------------------------------------

def processar(entrada, cfg: dict, progresso=None) -> dict:
    """Roda a cadeia inteira e devolve o resultado na memória (não grava).
    entrada = caminho da mix, ou lista de caminhos das stems."""
    p = progresso or (lambda pct, txt: None)
    cfg = completar(cfg)
    avisos: list[str] = []
    feito: list[str] = []
    if isinstance(entrada, (list, tuple)) and len(entrada) == 1:
        entrada = entrada[0]
    modo_stems = not isinstance(entrada, str)

    lista_stems = None
    if modo_stems:
        x, sr, lista_stems = carregar_stems(list(entrada), cfg["stems_ganhos"], cfg["stems_mudos"], avisos,
                                            lambda k, n: p(1 + int(5 * k / n), f"Somando as stems ({k}/{n})…"))
        ativas = [(nm, g) for nm, g in lista_stems if g is not None]
        mexidas = [f"{nm} {g:+.1f} dB" for nm, g in ativas if abs(g) >= 0.05]
        mudas = [nm for nm, g in lista_stems if g is None]
        t = f"Soma de {len(ativas)} stems"
        if mexidas:
            t += "; volumes: " + ", ".join(mexidas)
        if mudas:
            t += "; no mudo: " + ", ".join(mudas)
        feito.append(t + ".")
    else:
        p(2, "Lendo a mix…")
        x, sr = ler(entrada)
    estereo = x.shape[1] == 2
    p(6, "Analisando a mix…")
    antes = analisar(x, sr)
    if antes["lufs"] is None:
        raise ValueError("não consegui medir o loudness: o arquivo parece estar em silêncio")
    if antes["clipes"] > 10 and not modo_stems:
        avisos.append(f"A mix já chegou clipada ({antes['clipes']} trechos grudados no 0 dB). "
                      "Abaixe o master da DAW (Stereo Out no Logic, MASTER no REAPER) e exporte de novo: clip não tem conserto no master.")

    alvo = float(cfg["alvo_lufs"])
    curva_estilo = ESTILOS.get(cfg["estilo"], ESTILOS["geral"])[1]
    ref, xr, srr = None, None, None
    if cfg["destino"] == "referencia":
        if not cfg.get("referencia"):
            raise ValueError("escolha o arquivo da música de referência")
        p(10, "Analisando a referência…")
        xr, srr = ler(cfg["referencia"])
        ref = analisar(xr, srr)
        if ref["lufs"] is None:
            raise ValueError("a referência parece estar em silêncio")
        alvo = round(lim(ref["lufs"], -16.0, -7.0), 1)
        if abs(alvo - ref["lufs"]) > 0.05:
            avisos.append(f"A referência está em {ref['lufs']:.1f} LUFS; usei {alvo:.1f} (limite seguro entre -16 e -7).")
        curva_estilo = ref["bandas"]
    alvo = lim(alvo, LUFS_MIN, LUFS_MAX)
    teto = lim(float(cfg["teto_tp"]), TETO_MIN, TETO_MAX)

    # 1. passa-altas
    y = dsp.passa_altas(x, sr, 25.0)
    feito.append("Passa-altas suave em 25 Hz (tira sub inaudível que só gasta espaço).")

    # 2. stems
    if not modo_stems and any(abs(float(v or 0)) >= 0.05 for v in cfg["stems"].values()):
        p(12, "Rebalanceando voz / bateria / baixo…")
        y, g = stems.rebalancear(y, sr, cfg["stems"],
                                 lambda k, n: p(12 + int(14 * k / n), f"Rebalanceando voz / bateria / baixo ({k}/{n})…"))
        partes = [f"{stems.STEMS[k]} {v:+.1f} dB" for k, v in g.items() if abs(v) >= 0.05]
        feito.append("Rebalanceamento (aproximado): " + ", ".join(partes) + ".")

    # 3. match EQ
    usa_match = ref is not None and float(cfg["match_eq"]) > 0
    if usa_match:
        p(27, "Match EQ com a referência…")
        fc = dsp.centros_terco(min(sr, srr))
        cm = dsp.curva_terco(y, sr)[: len(fc)]
        cr = dsp.curva_terco(xr, srr)[: len(fc)]
        curva = dsp.curva_match(cm, cr, fc, float(cfg["match_eq"]) / 100)
        y = dsp.aplicar_curva(y, sr, fc, curva)
        top = np.argsort(-np.abs(curva))[:3]
        txt = ", ".join(f"{fc[i]:.0f} Hz {curva[i]:+.1f} dB" for i in sorted(top))
        feito.append(f"Match EQ com a referência ({float(cfg['match_eq']):.0f}%): maiores ajustes em {txt}.")

    # 4. graves e agudos
    if usa_match:
        grave_auto, agudo_auto = 0.0, 0.0  # o match EQ já cuida do equilíbrio
    else:
        grave_auto, agudo_auto = eq_automatico(antes["bandas"], curva_estilo)
    grave = grave_auto if cfg["grave"] is None else lim(float(cfg["grave"]), -LIMITE_EQ_DB, LIMITE_EQ_DB)
    agudo = agudo_auto if cfg["agudo"] is None else lim(float(cfg["agudo"]), -LIMITE_EQ_DB, LIMITE_EQ_DB)
    p(32, "Equalizando…")
    y = dsp.equalizar(y, sr, grave, agudo)
    feito.append(f"Graves (shelf 120 Hz): {grave:+.1f} dB" + (" (automático)" if cfg["grave"] is None else " (manual)"))
    feito.append(f"Agudos (shelf 8 kHz): {agudo:+.1f} dB" + (" (automático)" if cfg["agudo"] is None else " (manual)"))

    # 5. mid/side
    mid, side = cfg["ms_mid"], cfg["ms_side"]
    if any(abs(float(v)) >= 0.05 for v in list(mid.values()) + list(side.values())):
        y = dsp.eq_mid_side(y, sr, mid, side)
        t = (f"centro: graves {mid['grave']:+.1f}, presença {mid['presenca']:+.1f}, agudos {mid['agudo']:+.1f} dB")
        if estereo:
            t += f"; laterais: graves {side['grave']:+.1f}, agudos {side['agudo']:+.1f} dB"
        feito.append(f"EQ mid/side — {t}.")

    # 6. EQ dinâmico
    max_din = DINAMICO.get(cfg["dinamico"], DINAMICO["desligado"])[1]
    if max_din > 0 and cfg["dinamico_faixas"]:
        p(38, "EQ dinâmico…")
        y, info = dsp.eq_dinamico(y, sr, list(cfg["dinamico_faixas"]), max_din)
        for i in info:
            feito.append(f"EQ dinâmico — {i['nome']} em {i['freq']:.0f} Hz: atuou em {i['tempo_pct']:.0f}% do tempo, "
                         f"média {i['gr_media']:.1f} dB, máx. {i['gr_max']:.1f} dB.")

    # 7. multibanda
    if cfg["multibanda"] != "nenhuma":
        p(44, "Compressão multibanda…")
        y, info = dsp.multibanda(y, sr, cfg["multibanda"])
        if info:
            t = ", ".join(f"{i['nome'].lower()} {i['gr']:.1f} dB" for i in info)
            feito.append(f"Multibanda {dsp.MULTIBANDA[cfg['multibanda']][0].lower()} (cortes 150 Hz e 2,5 kHz): "
                         f"redução nas partes fortes — {t}.")

    # 8. cola
    comp_txt, ratio, gr_alvo = COMPRESSAO.get(cfg["compressao"], COMPRESSAO["leve"])
    if ratio:
        p(50, "Compressor de cola…")
        y, gr_comp = dsp.comprimir(y, sr, ratio, gr_alvo)
        feito.append(f"Compressão {comp_txt.lower()}: ~{gr_comp:.1f} dB de redução nas partes fortes.")

    # 9. estéreo
    if estereo:
        mono_hz = float(cfg["mono_grave_hz"]) if cfg["mono_grave"] else None
        largura = float(cfg["largura"]) / 100
        if mono_hz or abs(largura - 1) >= 0.01:
            y = dsp.imagem_estereo(y, sr, mono_hz, largura)
            t = []
            if mono_hz:
                t.append(f"grave em mono abaixo de {mono_hz:.0f} Hz")
            if abs(largura - 1) >= 0.01:
                t.append(f"largura {largura * 100:.0f}% acima de 300 Hz")
            feito.append("Estéreo: " + ", ".join(t) + ".")

    # 10. saturação
    if cfg["saturacao"]:
        p(54, "Saturando…")
        y = dsp.saturar(y)
        feito.append("Saturação leve (tipo fita).")

    # 11. soft clip
    modo = cfg["limitador"] if cfg["limitador"] in dsp.MODOS_LIMITADOR else "transparente"
    clip_db = dsp.MODOS_LIMITADOR[modo][3]
    if clip_db > 0:
        p(57, "Soft clip…")
        y = dsp.soft_clip(y, clip_db)
        feito.append(f"Soft clip: arredonda os {clip_db:.0f} dB mais altos dos picos antes do limitador.")

    # 12. limitador + loudness
    p(60, "Ajustando o loudness…")
    res = dsp.ajustar_loudness(y, sr, alvo, teto, modo, p)
    nome_lim = dsp.MODOS_LIMITADOR[modo][0].split(" (")[0].lower()
    if res["gr_max"] < 0.1:
        feito.append(f"Limitador ({nome_lim}): nem precisou atuar — o alvo foi atingido sem passar do teto.")
    else:
        feito.append(f"Limitador ({nome_lim}): redução média de {res['gr_media']:.1f} dB quando atua, "
                     f"máxima de {res['gr_max']:.1f} dB.")
    if res["gr_media"] > 4.0:
        avisos.append(f"O limitador está trabalhando muito (redução média de {res['gr_media']:.1f} dB): pode soar "
                      "esmagado. Se não gostar, tente um alvo de LUFS mais baixo, a multibanda ou o modo Soft clip.")
    if res["lufs"] is not None and abs(res["lufs"] - alvo) > 0.5:
        avisos.append(f"Não deu para chegar ao alvo sem passar do teto: ficou em {res['lufs']:.1f} LUFS.")

    p(94, "Medindo o resultado…")
    depois = analisar(res["audio"], sr)
    graficos = {
        "med_original": dsp.medicoes(x, sr), "med_master": dsp.medicoes(res["audio"], sr),
        "gr": dsp.gr_por_quadro(res.pop("ganho_limitador"), sr),
        "esp_antes": dsp.curva_espectro(x, sr), "esp_depois": dsp.curva_espectro(res["audio"], sr),
        "esp_ref": dsp.curva_espectro(xr, srr) if ref is not None else None,
    }
    if estereo and depois["correlacao"] is not None and depois["correlacao"] < 0.1:
        avisos.append(f"Correlação estéreo baixa ({depois['correlacao']:.2f}): pode perder coisas quando tocar em mono. "
                      "Diminua a largura.")
    r = {"entrada": entrada, "audio": res["audio"], "original": x, "taxa": sr, "stems": lista_stems,
         "antes": antes, "depois": depois, "referencia": ref, "alvo": alvo, "teto": teto,
         "destino": cfg["destino"], "feito": feito, "avisos": avisos, "cfg": cfg,
         "grave": grave, "agudo": agudo, "graficos": graficos}
    r["relatorio"] = relatorio(r)
    p(100, "Pronto para ouvir!")
    return r


def salvar(r: dict, saida: str | None = None) -> dict:
    """Grava o WAV 24 bits e o relatório ao lado da mix (ou das stems)."""
    saida = saida or caminho_saida(r["entrada"])
    sf.write(saida, r["audio"], r["taxa"], subtype="PCM_24")
    r["saida"] = saida
    r["relatorio"] = relatorio(r)
    caminho_rel = os.path.splitext(saida)[0] + "_relatorio.txt"
    with open(caminho_rel, "w", encoding="utf-8") as f:
        f.write(r["relatorio"])
    r["arquivo_relatorio"] = caminho_rel
    return r


def masterizar(entrada: str, cfg: dict, progresso=None) -> dict:
    """Processa e salva de uma vez (linha de comando e testes)."""
    return salvar(processar(entrada, cfg, progresso))


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _fmt(v, unid="", casas=1):
    return "—" if v is None else f"{v:.{casas}f}{unid}"


def relatorio(r: dict) -> str:
    a, d = r["antes"], r["depois"]
    din_a = None if a["curto_p90"] is None else a["curto_p90"] - a["curto_p10"]
    din_d = None if d["curto_p90"] is None else d["curto_p90"] - d["curto_p10"]
    destino = DESTINOS.get(r["destino"], ("Personalizado", None))[0]
    master = os.path.basename(r["saida"]) if r.get("saida") else "(ainda não salvo)"
    linhas = [
        "MASTER — relatório",
        "=" * 50,
        ("Stems:   " if r.get("stems") else "Mix:     ") + descricao_entrada(r["entrada"]),
        f"Master:  {master}  (WAV 24 bits, {r['taxa']} Hz)",
        f"Destino: {destino}",
        f"Alvo:    {r['alvo']:.1f} LUFS, teto {r['teto']:.1f} dBTP",
        "",
        "                 ANTES        DEPOIS",
        f"Loudness      {_fmt(a['lufs'], ' LUFS'):>11}   {_fmt(d['lufs'], ' LUFS'):>11}",
        f"True peak     {_fmt(a['true_peak_db'], ' dBTP'):>11}   {_fmt(d['true_peak_db'], ' dBTP'):>11}",
        f"Dinâmica*     {_fmt(din_a, ' dB'):>11}   {_fmt(din_d, ' dB'):>11}",
    ]
    if a["correlacao"] is not None:
        linhas.append(f"Correlação**  {_fmt(a['correlacao'], '', 2):>11}   {_fmt(d['correlacao'], '', 2):>11}")
    linhas.append("  * diferença entre as partes fortes e as fracas (loudness de 3 s).")
    if a["correlacao"] is not None:
        linhas.append("  ** estéreo: +1 = mono, perto de 0 = bem aberto, negativo = problema de fase.")
    linhas += ["", "O que foi feito:"] + [f"  • {t}" for t in r["feito"]]
    if r.get("referencia"):
        rf = r["referencia"]
        linhas += ["", f"Referência: {_fmt(rf['lufs'], ' LUFS')}, true peak {_fmt(rf['true_peak_db'], ' dBTP')}."]
    if r["avisos"]:
        linhas += ["", "Atenção:"] + [f"  • {t}" for t in r["avisos"]]
    linhas += ["", "Dica: use o A/B com 'Igualar volume' ligado: assim você compara o som, não só o 'mais alto'."]
    return "\n".join(linhas) + "\n"


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Master de uma mix (WAV/AIFF/FLAC).")
    ap.add_argument("entrada", nargs="+", help="a mix, ou várias stems")
    ap.add_argument("--destino", default="personalizado", choices=list(DESTINOS))
    ap.add_argument("--alvo", type=float, default=-14.0)
    ap.add_argument("--teto", type=float, default=-1.0)
    ap.add_argument("--estilo", default="geral", choices=list(ESTILOS))
    ap.add_argument("--referencia", default=None)
    ap.add_argument("--match", type=float, default=60)
    ap.add_argument("--grave", type=float, default=None)
    ap.add_argument("--agudo", type=float, default=None)
    ap.add_argument("--dinamico", default="desligado", choices=list(DINAMICO))
    ap.add_argument("--multibanda", default="nenhuma", choices=list(dsp.MULTIBANDA))
    ap.add_argument("--compressao", default="leve", choices=list(COMPRESSAO))
    ap.add_argument("--largura", type=float, default=100)
    ap.add_argument("--sem-mono-grave", action="store_true")
    ap.add_argument("--voz", type=float, default=0)
    ap.add_argument("--bateria", type=float, default=0)
    ap.add_argument("--baixo", type=float, default=0)
    ap.add_argument("--saturacao", action="store_true")
    ap.add_argument("--limitador", default="transparente", choices=list(dsp.MODOS_LIMITADOR))
    a = ap.parse_args()
    cfg = {"destino": a.destino, "alvo_lufs": a.alvo, "teto_tp": a.teto, "estilo": a.estilo,
           "referencia": a.referencia, "match_eq": a.match, "grave": a.grave, "agudo": a.agudo,
           "dinamico": a.dinamico, "multibanda": a.multibanda, "compressao": a.compressao,
           "largura": a.largura, "mono_grave": not a.sem_mono_grave,
           "stems": {"voz": a.voz, "bateria": a.bateria, "baixo": a.baixo},
           "saturacao": a.saturacao, "limitador": a.limitador}
    r = masterizar(a.entrada if len(a.entrada) > 1 else a.entrada[0], cfg, lambda pct, txt: print(f"[{pct:3d}%] {txt}"))
    print()
    print(r["relatorio"])


if __name__ == "__main__":
    _main()
