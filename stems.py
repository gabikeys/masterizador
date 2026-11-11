"""
stems.py — rebalanceamento APROXIMADO de voz, bateria e baixo numa mix pronta.

Não é separação por rede neural (como nos programas caros). É feito no
espectro (STFT), separando por características do som:
  - bateria: o que é curto/percussivo (separação harmônico/percussivo por
    filtro de mediana: sustentado = linha no tempo, ataque = linha na frequência);
  - baixo: o que é sustentado e grave (abaixo de ~160 Hz);
  - voz: o que é sustentado, está no CENTRO do estéreo e na região da voz
    (150 Hz a 9 kHz). Pega também outros instrumentos centrais e sustentados.
As máscaras não se sobrepõem, então o que não é nenhum dos três fica igual.
Funciona bem para ajustes pequenos (±2 a 3 dB).
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from dsp import lim, LIMITE_EQ_DB

STEMS = {"voz": "Voz (centro)", "bateria": "Bateria / ataques", "baixo": "Baixo / grave"}


def _por_trechos(x: np.ndarray, sr: int, func, seg_s: float = 20.0, sobre_s: float = 0.5, progresso=None) -> np.ndarray:
    """Processa em trechos com sobreposição e transição suave (poupa memória)."""
    n = len(x)
    seg, sobre = int(seg_s * sr), int(sobre_s * sr)
    saida = np.zeros_like(x)
    peso = np.zeros(n)
    inicios = list(range(0, n, seg))
    for k, a in enumerate(inicios):
        a0, b0 = max(0, a - sobre), min(n, a + seg + sobre)
        y = func(x[a0:b0])
        w = np.ones(b0 - a0)
        L = min(2 * sobre, len(w))  # o último trecho pode ser curto
        if a0 > 0:
            w[:L] = np.minimum(w[:L], np.linspace(1e-3, 1, L))
        if b0 < n:
            w[-L:] = np.minimum(w[-L:], np.linspace(1, 1e-3, L))
        saida[a0:b0] += y * w[:, None]
        peso[a0:b0] += w
        if progresso:
            progresso(k + 1, len(inicios))
    return saida / np.maximum(peso, 1e-9)[:, None]


def mascaras(Z: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z: STFT (canais, bins, quadros). Devolve máscaras (voz, bateria, baixo)."""
    mag = np.abs(Z.mean(axis=0)) if Z.shape[0] > 1 else np.abs(Z[0])
    sustentado = median_filter(mag, size=(1, 17), mode="nearest")
    ataque = median_filter(mag, size=(17, 1), mode="nearest")
    h2, p2 = sustentado ** 2, ataque ** 2
    m_harm = h2 / (h2 + p2 + 1e-20)
    m_perc = 1.0 - m_harm

    ff = np.maximum(f, 1.0)[:, None]
    w_grave = 1.0 / (1.0 + (ff / 160.0) ** 4)
    w_voz = (1.0 / (1.0 + (150.0 / ff) ** 4)) * (1.0 / (1.0 + (ff / 9000.0) ** 4))
    if Z.shape[0] > 1:
        L, R = np.abs(Z[0]), np.abs(Z[1])
        mesma_amplitude = 1.0 - np.abs(L - R) / (L + R + 1e-20)
        mesma_fase = np.maximum(0.0, np.real(Z[0] * np.conj(Z[1])) / (L * R + 1e-20))
        centro = (mesma_amplitude * mesma_fase) ** 2
    else:
        centro = 1.0
    m_baixo = m_harm * w_grave
    m_voz = m_harm * (1.0 - w_grave) * w_voz * centro
    return m_voz, m_perc, m_baixo


def rebalancear(x: np.ndarray, sr: int, ganhos_db: dict, progresso=None) -> tuple[np.ndarray, dict]:
    """ganhos_db = {"voz": dB, "bateria": dB, "baixo": dB} (cada um até ±6 dB)."""
    g = {k: lim(float(ganhos_db.get(k, 0.0) or 0.0), -LIMITE_EQ_DB, LIMITE_EQ_DB) for k in STEMS}
    if all(abs(v) < 0.05 for v in g.values()):
        return x, {}
    gl = {k: 10 ** (v / 20) - 1.0 for k, v in g.items()}
    nfft = 4096 if sr <= 50000 else 8192
    hop = nfft // 4

    def processar(trecho):
        f, _, Z = signal.stft(trecho.T, fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
        m_voz, m_bat, m_baixo = mascaras(Z, f)
        G = 1.0 + gl["voz"] * m_voz + gl["bateria"] * m_bat + gl["baixo"] * m_baixo
        _, y = signal.istft(Z * G[None], fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
        y = y.T[: len(trecho)]
        if len(y) < len(trecho):
            y = np.pad(y, ((0, len(trecho) - len(y)), (0, 0)))
        return y

    y = _por_trechos(x, sr, processar, progresso=progresso)
    return y, g
