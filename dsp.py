"""
dsp.py — blocos de processamento e medidas do masterizador.

Tudo trabalha com áudio float64 no formato (amostras, canais), 1 ou 2 canais.
Só usa numpy e scipy (o filtro K do loudness vem do pyloudnorm).
"""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

LIMITE_EQ_DB = 6.0        # segurança: nenhum EQ passa de ±6 dB
LIMITE_RATIO = 8.0        # segurança: ratio nunca passa de 8:1


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def lim(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def db_pot(p: float) -> float:
    return 10 * math.log10(p) if p > 1e-20 else -200.0


def db_amp(a: float) -> float:
    return 20 * math.log10(a) if a > 1e-10 else -200.0


def abs_max_canais(x: np.ndarray) -> np.ndarray:
    """|x| máximo entre os canais, amostra a amostra."""
    a = np.abs(x[:, 0])
    for c in range(1, x.shape[1]):
        a = np.maximum(a, np.abs(x[:, c]))
    return a


def mono(x: np.ndarray) -> np.ndarray:
    return x[:, 0].copy() if x.shape[1] == 1 else (x[:, 0] + x[:, 1]) / 2


def potencia_canais(x: np.ndarray) -> np.ndarray:
    """Média de x² entre os canais, amostra a amostra."""
    p = np.ascontiguousarray(x[:, 0]) ** 2
    for c in range(1, x.shape[1]):
        p = p + np.ascontiguousarray(x[:, c]) ** 2
    return p / x.shape[1]


def filtrar(sos: np.ndarray, canal: np.ndarray, zero_fase: bool = False) -> np.ndarray:
    """Filtro IIR com proteção contra números subnormais: em silêncio digital
    o estado do filtro decai até valores minúsculos que o processador calcula
    centenas de vezes mais devagar. Um sinal alternado de -400 dB evita isso.
    zero_fase=True filtra ida e volta (sem defasagem: dá para subtrair do
    original e separar bandas sem 'buraco' no cruzamento)."""
    anti = np.full(len(canal), 1e-20)
    anti[1::2] = -1e-20
    c = np.ascontiguousarray(canal) + anti
    return signal.sosfiltfilt(sos, c) if zero_fase else signal.sosfilt(sos, c)


def por_canal(x: np.ndarray, func) -> np.ndarray:
    y = np.empty_like(x)
    for c in range(x.shape[1]):
        y[:, c] = func(np.ascontiguousarray(x[:, c]))
    return y


def vezes(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    for c in range(x.shape[1]):
        y[:, c] = x[:, c] * g
    return y


def por_pedacos(canal: np.ndarray, func, passo: int = 1 << 18, borda: int = 256) -> np.ndarray:
    """Aplica func em pedaços (com bordas) de um canal, para poupar memória."""
    saida = np.empty_like(canal)
    n = len(canal)
    for a in range(0, n, passo):
        b = min(n, a + passo)
        a0, b0 = max(0, a - borda), min(n, b + borda)
        y = func(canal[a0:b0])
        saida[a:b] = y[a - a0: a - a0 + (b - a)]
    return saida


def para_mid_side(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2


def de_mid_side(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.stack([m + s, m - s], axis=1)


def nivel_blocos(p: np.ndarray, blk: int) -> np.ndarray:
    """Nível (dB) de uma potência p em blocos de blk amostras."""
    nb = len(p) // blk
    # max(0): médias móveis podem dar -1e-18 no silêncio (arredondamento)
    return 10 * np.log10(np.maximum(p[: nb * blk].reshape(nb, blk).mean(axis=1), 0.0) + 1e-20)


def suavizar(alvo: np.ndarray, ataque: float, release: float) -> np.ndarray:
    """Segue 'alvo' (redução em dB, >= 0) com ataque/release (coeficientes 0..1)."""
    saida = np.empty(len(alvo))
    atual = 0.0
    for i, t in enumerate(alvo.tolist()):
        c = ataque if t > atual else release
        atual = t + (atual - t) * c
        saida[i] = atual
    return saida


def blocos_para_amostras(valores: np.ndarray, blk: int, n: int) -> np.ndarray:
    centros = (np.arange(len(valores)) + 0.5) * blk
    return np.interp(np.arange(n), centros, valores)


def coef(ms: float, blk: int, sr: int) -> float:
    return math.exp(-blk / sr / (ms / 1000))


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

_FILTROS_K: dict = {}


def _filtro_k(sr: int) -> np.ndarray:
    if sr not in _FILTROS_K:
        medidor = pyln.Meter(sr)
        _FILTROS_K[sr] = np.vstack([signal.tf2sos(e.b, e.a) for e in medidor._filters.values()])
    return _FILTROS_K[sr]


def _media_janelas(z: np.ndarray, janela: int, passo: int) -> np.ndarray:
    if len(z) < janela:
        return np.array([])
    c = np.concatenate([[0.0], np.cumsum(z, dtype=np.float64)])
    ini = np.arange(0, len(z) - janela + 1, passo)
    return (c[ini + janela] - c[ini]) / janela


def loudness(x: np.ndarray, sr: int) -> dict:
    """LUFS integrado (BS.1770, com gates) e de curto prazo (3 s)."""
    r = {"lufs": None, "curto_p10": None, "curto_p90": None, "curto_max": None}
    sos = _filtro_k(sr)
    z = np.zeros(len(x))
    for ch in range(x.shape[1]):
        z += filtrar(sos, x[:, ch]) ** 2

    def lufs(ms):
        return -0.691 + 10 * np.log10(np.maximum(ms, 1e-20))

    blocos = _media_janelas(z, int(0.4 * sr), int(0.1 * sr))
    if len(blocos):
        ok = lufs(blocos) > -70
        if np.any(ok):
            relativo = lufs(np.mean(blocos[ok])) - 10
            ok2 = ok & (lufs(blocos) > relativo)
            if np.any(ok2):
                r["lufs"] = float(lufs(np.mean(blocos[ok2])))
    curtos = lufs(_media_janelas(z, 3 * sr, sr))
    curtos = curtos[curtos > -70]
    if len(curtos):
        r["curto_p10"] = float(np.percentile(curtos, 10))
        r["curto_p90"] = float(np.percentile(curtos, 90))
        r["curto_max"] = float(np.max(curtos))
    return r


def true_peak(x: np.ndarray) -> float:
    """True peak (dBTP) por sobreamostragem 4x, só nos trechos perto do pico."""
    a = abs_max_canais(x)
    pico_amostra = float(np.max(a)) if len(a) else 0.0
    if pico_amostra <= 0:
        return -200.0
    bloco = 4096
    n = len(a) // bloco
    maximos = a[: n * bloco].reshape(n, bloco).max(axis=1) if n else np.array([])
    candidatos = list(np.flatnonzero(maximos >= pico_amostra * 10 ** (-3 / 20)))
    if len(a) > n * bloco:
        candidatos.append(n)
    pico = pico_amostra
    for b in candidatos[:3000]:
        ini, fim = max(0, b * bloco - 64), min(len(x), (b + 1) * bloco + 64)
        for c in range(x.shape[1]):
            y = signal.resample_poly(np.ascontiguousarray(x[ini:fim, c]), 4, 1)
            pico = max(pico, float(np.max(np.abs(y))))
    return db_amp(pico)


def correlacao(x: np.ndarray) -> float | None:
    """Correlação L/R (+1 = mono, 0 = bem aberto, negativo = problema de fase)."""
    if x.shape[1] < 2:
        return None
    l, r = x[:, 0], x[:, 1]
    d = math.sqrt(float(np.dot(l, l)) * float(np.dot(r, r)))
    return float(np.dot(l, r)) / d if d > 0 else None


def espectro(x: np.ndarray, sr: int, resolucao: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Espectro médio (Welch) com pedaços espalhados da música (rápido)."""
    m = mono(x)
    if len(m) > 90 * sr:
        tam = 2 * sr
        inicios = np.linspace(0, len(m) - tam, 45).astype(int)
        m = np.concatenate([m[i:i + tam] for i in inicios])
    nper = min(len(m), 2 ** int(round(math.log2(resolucao * sr / 44100))))
    return signal.welch(m, fs=sr, nperseg=nper, noverlap=nper // 2)


PASSO_MEDIDOR = 0.05  # 50 ms por quadro dos medidores


def medicoes(x: np.ndarray, sr: int, passo: float = PASSO_MEDIDOR) -> dict:
    """Para os medidores da tela: pico por canal (dB) e loudness momentâneo
    (LUFS, janela de 400 ms) a cada 'passo' segundos."""
    blk = max(1, int(sr * passo))
    nb = len(x) // blk
    picos = np.empty((nb, x.shape[1]))
    for c in range(x.shape[1]):
        a = np.abs(np.ascontiguousarray(x[: nb * blk, c])).reshape(nb, blk).max(axis=1)
        picos[:, c] = 20 * np.log10(np.maximum(a, 1e-6))
    sos = _filtro_k(sr)
    z = np.zeros(len(x))
    for c in range(x.shape[1]):
        z += filtrar(sos, x[:, c]) ** 2
    jan = int(0.4 * sr)
    acum = np.concatenate([[0.0], np.cumsum(z)])
    fins = np.minimum((np.arange(nb) + 1) * blk, len(z))
    inis = np.maximum(fins - jan, 0)
    media = (acum[fins] - acum[inis]) / np.maximum(fins - inis, 1)
    lufs_m = -0.691 + 10 * np.log10(np.maximum(media, 1e-12))
    return {"passo": passo, "pico": picos.astype(np.float32), "lufs_m": lufs_m.astype(np.float32)}


def gr_por_quadro(g: np.ndarray, sr: int, passo: float = PASSO_MEDIDOR) -> np.ndarray:
    """Maior redução do limitador (dB, positiva) em cada quadro dos medidores."""
    blk = max(1, int(sr * passo))
    nb = len(g) // blk
    m = g[: nb * blk].reshape(nb, blk).min(axis=1)
    return (-20 * np.log10(np.maximum(m, 1e-6))).astype(np.float32)


def curva_espectro(x: np.ndarray, sr: int, pontos: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Espectro médio suavizado (1/6 de oitava) numa grade logarítmica de 20 Hz a 20 kHz."""
    f, p = espectro(x, sr, resolucao=16384)
    grade = np.geomspace(20, min(20000, sr * 0.48), pontos)
    pdb = np.empty(pontos)
    for i, c in enumerate(grade):
        sel = (f >= c * 2 ** (-1 / 12)) & (f <= c * 2 ** (1 / 12))
        if not np.any(sel):
            sel = np.abs(f - c) == np.min(np.abs(f - c))
        pdb[i] = 10 * np.log10(np.mean(p[sel]) + 1e-30)
    return grade, pdb


# ---------------------------------------------------------------------------
# EQ
# ---------------------------------------------------------------------------

def biquad(sr: int, tipo: str, freq: float, ganho_db: float = 0.0, q: float = 0.707) -> np.ndarray:
    """Biquads do 'Audio EQ Cookbook': 'grave' (shelf), 'agudo' (shelf), 'sino'."""
    A = 10 ** (ganho_db / 40)
    w0 = 2 * math.pi * freq / sr
    cw, sw = math.cos(w0), math.sin(w0)
    if tipo == "sino":
        alfa = sw / (2 * q)
        b = [1 + alfa * A, -2 * cw, 1 - alfa * A]
        a = [1 + alfa / A, -2 * cw, 1 - alfa / A]
    else:
        alfa = sw / 2 * math.sqrt(2)
        raiz = 2 * math.sqrt(A) * alfa
        if tipo == "grave":
            b = [A * ((A + 1) - (A - 1) * cw + raiz), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - raiz)]
            a = [(A + 1) + (A - 1) * cw + raiz, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - raiz]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + raiz), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - raiz)]
            a = [(A + 1) - (A - 1) * cw + raiz, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - raiz]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def eq_sos(sr: int, bandas: list[tuple]) -> np.ndarray | None:
    """bandas = [(tipo, freq, ganho_db, q), ...]; ignora as neutras."""
    sos = []
    for tipo, freq, ganho, q in bandas:
        ganho = lim(ganho, -LIMITE_EQ_DB, LIMITE_EQ_DB)
        if abs(ganho) >= 0.05 and freq < sr * 0.45:
            sos.append(biquad(sr, tipo, freq, ganho, q))
    return np.vstack(sos) if sos else None


def passa_altas(x: np.ndarray, sr: int, freq: float = 25.0) -> np.ndarray:
    sos = signal.butter(2, freq, "highpass", fs=sr, output="sos")
    return por_canal(x, lambda c: filtrar(sos, c))


def equalizar(x: np.ndarray, sr: int, grave_db: float, agudo_db: float) -> np.ndarray:
    """Shelf de grave (120 Hz) e de agudo (8 kHz)."""
    sos = eq_sos(sr, [("grave", 120, grave_db, 0.7), ("agudo", 8000, agudo_db, 0.7)])
    return x if sos is None else por_canal(x, lambda c: filtrar(sos, c))


def eq_mid_side(x: np.ndarray, sr: int, mid: dict, side: dict) -> np.ndarray:
    """EQ separado para o centro (mid) e as laterais (side).
    mid: grave (shelf 120), presenca (sino 3 kHz), agudo (shelf 8 k);
    side: grave (shelf 200), agudo (shelf 8 k). Em arquivo mono, só o mid."""
    sos_m = eq_sos(sr, [("grave", 120, mid.get("grave", 0), 0.7), ("sino", 3000, mid.get("presenca", 0), 0.9),
                        ("agudo", 8000, mid.get("agudo", 0), 0.7)])
    if x.shape[1] < 2:
        return x if sos_m is None else por_canal(x, lambda c: filtrar(sos_m, c))
    sos_s = eq_sos(sr, [("grave", 200, side.get("grave", 0), 0.7), ("agudo", 8000, side.get("agudo", 0), 0.7)])
    if sos_m is None and sos_s is None:
        return x
    m, s = para_mid_side(x)
    if sos_m is not None:
        m = filtrar(sos_m, m)
    if sos_s is not None:
        s = filtrar(sos_s, s)
    return de_mid_side(m, s)


# --- Match EQ ---------------------------------------------------------------

def centros_terco(sr: int) -> np.ndarray:
    fc = 1000 * 2 ** (np.arange(-16, 15) / 3)  # ~20 Hz a ~20 kHz
    return fc[(fc >= 25) & (fc <= min(16000, sr * 0.45))]


def curva_terco(x: np.ndarray, sr: int) -> np.ndarray:
    """Densidade de energia (dB) em bandas de 1/3 de oitava."""
    f, p = espectro(x, sr, resolucao=16384)
    fc = centros_terco(sr)
    r = []
    for c in fc:
        sel = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        r.append(db_pot(float(np.mean(p[sel]))) if np.any(sel) else -200.0)
    return np.array(r)


def curva_match(curva_mix: np.ndarray, curva_ref: np.ndarray, fc: np.ndarray, quanto: float) -> np.ndarray:
    """Diferença suavizada (≈1 oitava) entre a referência e a mix, sem mudar o
    volume geral (alinha na média de 100 Hz a 8 kHz), × quanto, limitada a ±6 dB."""
    d = curva_ref - curva_mix
    meio = (fc >= 100) & (fc <= 8000)
    d = d - np.mean(d[meio])
    d = np.convolve(np.pad(d, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    d = np.clip(d * quanto, -LIMITE_EQ_DB, LIMITE_EQ_DB)
    d[(fc < 35) | (fc > 14000)] *= 0.5  # extremos com mais cuidado
    return d


def aplicar_curva(x: np.ndarray, sr: int, fc: np.ndarray, ganhos_db: np.ndarray) -> np.ndarray:
    """Aplica uma curva de EQ com FIR de fase linear (não mexe na fase)."""
    if np.max(np.abs(ganhos_db)) < 0.05:
        return x
    nyq = sr / 2
    freqs = np.concatenate([[0.0], fc, [nyq]])
    g = 10 ** (np.concatenate([[ganhos_db[0]], ganhos_db, [ganhos_db[-1]]]) / 20)
    taps = 2 ** int(round(math.log2(4096 * sr / 44100))) + 1
    h = signal.firwin2(taps, freqs, g, fs=sr)
    atraso = (taps - 1) // 2
    return por_canal(x, lambda c: signal.oaconvolve(c, h)[atraso: atraso + len(c)])


# --- EQ dinâmico -------------------------------------------------------------

FAIXAS_DINAMICAS = {  # chave: (texto, Hz mínimo, Hz máximo)
    "embolado": ("Embolado", 180, 500),
    "aspereza": ("Aspereza", 2000, 5000),
    "sibilancia": ("Sibilância", 5000, 10000),
}


def achar_ressonancia(f: np.ndarray, p: np.ndarray, a: float, b: float) -> float:
    """Frequência que mais 'salta' do espectro dentro de [a, b]: compara a
    média de 1/6 de oitava com a tendência de 1 oitava ao redor."""
    pdb = 10 * np.log10(p + 1e-30)
    melhor, fmelhor = -1e9, math.sqrt(a * b)
    for c in np.geomspace(a, b, 24):
        perto = (f >= c * 2 ** (-1 / 12)) & (f <= c * 2 ** (1 / 12))
        volta = (f >= c * 2 ** (-1 / 2)) & (f <= c * 2 ** (1 / 2))
        if np.any(perto) and np.any(volta):
            v = float(np.mean(pdb[perto]) - np.mean(pdb[volta]))
            if v > melhor:
                melhor, fmelhor = v, float(c)
    return fmelhor


def eq_dinamico(x: np.ndarray, sr: int, faixas: list[str], max_db: float) -> tuple[np.ndarray, list[dict]]:
    """Para cada faixa: acha a frequência problemática e abaixa SÓ essa região,
    SÓ nos momentos em que ela passa do normal da música (ratio 3:1, até max_db).
    A banda é separada com filtro de fase zero, então y = x + (g-1)·banda."""
    max_db = lim(max_db, 0.0, LIMITE_EQ_DB)
    info = []
    if max_db <= 0 or not faixas:
        return x, info
    f, p = espectro(x, sr)
    blk = max(1, int(sr * 0.005))
    for chave in faixas:
        nome, a, b = FAIXAS_DINAMICAS[chave]
        if b >= sr * 0.45:
            continue
        fc = achar_ressonancia(f, p, a, b)
        sos = signal.butter(2, [fc * 2 ** (-1 / 3), fc * 2 ** (1 / 3)], "bandpass", fs=sr, output="sos")
        banda = por_canal(x, lambda c: filtrar(sos, c, zero_fase=True))
        janela = max(1, int(sr * 0.02))  # detector com média de 20 ms (5 ms oscila demais)
        nb_ = nivel_blocos(uniform_filter1d(potencia_canais(banda), janela), blk)
        nt = nivel_blocos(uniform_filter1d(potencia_canais(x), janela), blk)
        if len(nt) < 20:
            continue
        ativos = nt > max(float(nt.max()) - 45, -70)
        if not np.any(ativos):
            continue
        rel = nb_ - nt
        # "normal" = o que a faixa tem em 70% do tempo; atua 3 dB acima disso
        normal = float(np.percentile(rel[ativos], 70)) + 3.0
        alvo = np.minimum(max_db, np.maximum(0.0, rel - normal) * (1 - 1 / 3))
        alvo[~ativos] = 0.0
        gr = suavizar(alvo, coef(5, blk, sr), coef(80, blk, sr))
        g = 10 ** (-blocos_para_amostras(gr, blk, len(x)) / 20)
        x = x + vezes(banda, g - 1.0)
        atua = gr > 0.2
        info.append({"nome": nome, "freq": fc, "gr_max": float(gr.max()),
                     "gr_media": float(np.mean(gr[atua])) if np.any(atua) else 0.0,
                     "tempo_pct": float(100 * np.mean(atua))})
    return x, info


# ---------------------------------------------------------------------------
# Dinâmica
# ---------------------------------------------------------------------------

def comprimir(x: np.ndarray, sr: int, ratio: float, gr_alvo: float,
              ataque_ms: float = 30.0, release_ms: float = 200.0) -> tuple[np.ndarray, float]:
    """Compressor com detector RMS (blocos de 10 ms), estéreo ligado. O threshold
    é escolhido sozinho para dar ~gr_alvo dB de redução média nas partes fortes
    (os 30% mais altos)."""
    ratio = lim(ratio, 1.0, LIMITE_RATIO)
    blk = max(1, int(sr * 0.01))
    if len(x) // blk < 20 or gr_alvo <= 0:
        return x, 0.0
    nivel = nivel_blocos(potencia_canais(x), blk)
    ativos = nivel > max(float(nivel.max()) - 40, -70)
    if not np.any(ativos):
        return x, 0.0
    fortes = nivel >= np.percentile(nivel[ativos], 70)
    inclinacao = 1 - 1 / ratio
    lo, hi = float(nivel.max()) - 40, float(nivel.max())
    for _ in range(40):  # busca binária do threshold
        meio = (lo + hi) / 2
        if np.mean(np.maximum(0.0, nivel[fortes] - meio)) * inclinacao > gr_alvo:
            lo = meio
        else:
            hi = meio
    limiar = (lo + hi) / 2
    ca, cr = coef(ataque_ms, blk, sr), coef(release_ms, blk, sr)
    for _ in range(4):
        # o ataque/release muda a redução real; corrige o threshold até bater
        gr = suavizar(np.maximum(0.0, nivel - limiar) * inclinacao, ca, cr)
        erro = float(np.mean(gr[fortes])) - gr_alvo
        if abs(erro) < 0.15:
            break
        limiar += erro / inclinacao
    g = 10 ** (-blocos_para_amostras(gr, blk, len(x)) / 20)
    return vezes(x, g), float(np.mean(gr[fortes]))


MULTIBANDA_CORTES = (150.0, 2500.0)
MULTIBANDA = {  # chave: (texto, ratio, redução média desejada por banda)
    "nenhuma": ("Desligada", None, 0.0),
    "leve": ("Leve", 2.0, 1.5),
    "media": ("Média", 3.0, 3.0),
}


def multibanda(x: np.ndarray, sr: int, intensidade: str) -> tuple[np.ndarray, list[dict]]:
    """Compressor de 3 bandas (grave < 150 Hz < médio < 2,5 kHz < agudo).
    As bandas são separadas com filtros de fase zero e por subtração, então
    somadas de volta dão exatamente o original (sem coloração no cruzamento)."""
    _, ratio, gr = MULTIBANDA.get(intensidade, MULTIBANDA["nenhuma"])
    if not ratio:
        return x, []
    f1, f2 = MULTIBANDA_CORTES
    sos1 = signal.butter(2, f1, "lowpass", fs=sr, output="sos")
    sos2 = signal.butter(2, f2, "lowpass", fs=sr, output="sos")
    grave = por_canal(x, lambda c: filtrar(sos1, c, True))
    resto = x - grave
    medio = por_canal(resto, lambda c: filtrar(sos2, c, True))
    agudo = resto - medio
    saida, info = 0.0, []
    for nome, banda, atk, rel in (("Graves", grave, 30, 200), ("Médios", medio, 15, 120), ("Agudos", agudo, 5, 80)):
        y, g = comprimir(banda, sr, ratio, gr, atk, rel)
        saida = saida + y
        info.append({"nome": nome, "gr": g})
    return saida, info


def saturar(x: np.ndarray, quanto: float = 0.3) -> np.ndarray:
    """Saturação leve tipo fita (tanh) com sobreamostragem 2x."""
    pico = float(np.max(abs_max_canais(x))) or 1.0
    drive = 1.5

    def curva(trecho):
        up = signal.resample_poly(trecho / pico, 2, 1)
        sat = np.tanh(drive * up) / math.tanh(drive)
        up = up + quanto * (sat - up)
        return signal.resample_poly(up, 1, 2)[: len(trecho)] * pico

    return por_canal(x, lambda c: por_pedacos(c, curva))


def soft_clip(x: np.ndarray, quanto_db: float) -> np.ndarray:
    """Arredonda os picos mais altos (os últimos quanto_db dB) antes do limitador,
    com sobreamostragem 2x. Os ataques ficam com mais 'pegada' e o limitador
    trabalha menos."""
    if quanto_db <= 0:
        return x
    pico = float(np.max(abs_max_canais(x))) or 1.0
    teto = pico * 10 ** (-quanto_db / 20)
    joelho = teto * 10 ** (-1.5 / 20)
    faixa = teto - joelho

    def curva(trecho):
        up = signal.resample_poly(trecho, 2, 1)
        a = np.abs(up)
        acima = a > joelho
        a[acima] = joelho + faixa * np.tanh((a[acima] - joelho) / faixa)
        return signal.resample_poly(np.sign(up) * a, 1, 2)[: len(trecho)]

    return por_canal(x, lambda c: por_pedacos(c, curva))


# ---------------------------------------------------------------------------
# Estéreo
# ---------------------------------------------------------------------------

def imagem_estereo(x: np.ndarray, sr: int, mono_grave_hz: float | None, largura: float) -> np.ndarray:
    """Grave em mono abaixo de mono_grave_hz e largura (0,5 a 1,5) acima de 300 Hz.
    Mexe só no 'side' (diferença L-R); o centro fica intacto."""
    if x.shape[1] < 2:
        return x
    largura = lim(largura, 0.5, 1.5)
    if not mono_grave_hz and abs(largura - 1.0) < 0.01:
        return x
    m, s = para_mid_side(x)
    if mono_grave_hz:
        sos = signal.butter(2, mono_grave_hz, "lowpass", fs=sr, output="sos")
        s = s - filtrar(sos, s, zero_fase=True)
    if abs(largura - 1.0) >= 0.01:
        sos = signal.butter(2, 300, "lowpass", fs=sr, output="sos")
        baixo = filtrar(sos, s, zero_fase=True)
        s = baixo + (s - baixo) * largura
    return de_mid_side(m, s)


# ---------------------------------------------------------------------------
# Limitador
# ---------------------------------------------------------------------------

MODOS_LIMITADOR = {  # chave: (texto, release ms, antecipação ms, soft clip dB)
    "transparente": ("Transparente", 80.0, 1.5, 0.0),
    "pegada": ("Pegada (mais alto, mais agressivo)", 30.0, 1.0, 0.0),
    "softclip": ("Soft clip + limitador (ataques mais vivos)", 80.0, 1.5, 2.0),
}


def envelope_true_peak(x: np.ndarray) -> np.ndarray:
    """Pico entre amostras (4x) em cada amostra, maior valor entre os canais."""
    n = len(x)
    e = abs_max_canais(x)
    passo, borda = 1 << 18, 128
    for c in range(x.shape[1]):
        canal = np.ascontiguousarray(x[:, c])
        for a in range(0, n, passo):
            b = min(n, a + passo)
            a0, b0 = max(0, a - borda), min(n, b + borda)
            up = np.abs(signal.resample_poly(canal[a0:b0], 4, 1))
            m = up[(a - a0) * 4:(b - a0) * 4].reshape(-1, 4).max(axis=1)
            np.maximum(e[a:b], m, out=e[a:b])
    return e


def limitar(x: np.ndarray, env: np.ndarray, sr: int, ganho: float, teto: float,
            release_ms: float = 80.0, antecipa_ms: float = 1.5) -> tuple[np.ndarray, float, float]:
    """Limitador com antecipação (o arquivo inteiro está na memória):
      1. ganho necessário em cada amostra = teto / (pico entre amostras × ganho);
      2. release suave (sobe devagar, desce na hora), em blocos de 32 amostras;
      3. mínimo numa janela de 2L+1 e média numa janela de L: o ganho desce
         suave ANTES do pico e nunca fica acima do necessário.
    Devolve (áudio, redução máxima em dB, redução média quando atua, ganho por amostra)."""
    n = len(x)
    r = np.minimum(1.0, teto / np.maximum(ganho * env, 1e-12))
    B = 32
    nb = -(-n // B)
    rb = np.pad(r, (0, nb * B - n), constant_values=1.0).reshape(nb, B).min(axis=1)
    rdb = (20 * np.log10(rb)).tolist()
    k = math.exp(-B / sr / (release_ms / 1000))
    suave = np.empty(nb)
    atual = 0.0
    for i, v in enumerate(rdb):
        atual *= k
        if v < atual:
            atual = v
        suave[i] = atual
    rs = np.repeat(10 ** (suave / 20), B)[:n]
    L = max(3, int(sr * antecipa_ms / 1000)) | 1
    h = minimum_filter1d(rs, size=2 * L + 1, mode="nearest")
    g = uniform_filter1d(h, size=L, mode="nearest")
    gr_db = -20 * np.log10(np.maximum(g, 1e-10))
    atuando = gr_db > 0.1
    gr_media = float(np.mean(gr_db[atuando])) if np.any(atuando) else 0.0
    return vezes(x, ganho * g), max(0.0, float(gr_db.max())), gr_media, g


def ajustar_loudness(x: np.ndarray, sr: int, alvo: float, teto_db: float, modo: str = "transparente",
                     progresso=None, pct=(60, 92)) -> dict:
    """Procura o ganho de entrada do limitador que dá o LUFS alvo; confere o
    true peak no fim e, se passar do teto, abaixa o teto interno e refaz."""
    _, release, antecipa, _ = MODOS_LIMITADOR.get(modo, MODOS_LIMITADOR["transparente"])
    env = envelope_true_peak(x)
    l0 = loudness(x, sr)["lufs"]
    if l0 is None:
        raise ValueError("não consegui medir o loudness: o arquivo parece estar em silêncio")
    teto_int = teto_db - 0.2
    ganho_db = alvo - l0
    melhor = None
    passada = 0
    for _rodada in range(3):
        historico = []
        for _ in range(10):
            passada += 1
            if progresso:
                progresso(min(pct[1], pct[0] + passada * 4), f"Ajustando o loudness (passada {passada})…")
            y, gr_max, gr_media, g = limitar(x, env, sr, 10 ** (ganho_db / 20), 10 ** (teto_int / 20),
                                             release, antecipa)
            l = loudness(y, sr)["lufs"]
            historico.append((ganho_db, l))
            melhor = {"audio": y, "lufs": l, "ganho_db": ganho_db, "gr_max": gr_max, "gr_media": gr_media,
                      "ganho_limitador": g}
            erro = alvo - l
            if abs(erro) <= 0.1:
                break
            incl = 1.0
            if len(historico) >= 2:
                (g1, l1), (g2, l2) = historico[-2], historico[-1]
                if abs(g2 - g1) > 0.01:
                    incl = lim((l2 - l1) / (g2 - g1), 0.1, 1.0)
            ganho_db = min(ganho_db + erro / incl, 36.0)
        tp = true_peak(melhor["audio"])
        melhor["true_peak_db"] = tp
        if tp <= teto_db + 0.05:
            break
        teto_int -= (tp - teto_db) + 0.05
    return melhor
