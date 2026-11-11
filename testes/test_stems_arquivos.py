"""Testes do master a partir de stems separadas. Rode: python testes/test_stems_arquivos.py"""

import math
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dsp  # noqa: E402
import motor  # noqa: E402

falhas = 0


def checar(cond, texto):
    global falhas
    print(("ok   " if cond else "FALHA ") + texto)
    if not cond:
        falhas += 1


def db_banda(x, sr, a, b):
    sos = signal.butter(4, [a, b], "bandpass", fs=sr, output="sos")
    return 10 * math.log10(float(np.mean(signal.sosfilt(sos, x, axis=0) ** 2)) + 1e-30)


sr = 48000
seg = 20
t = np.arange(sr * seg) / sr
rng = np.random.default_rng(5)
baixo = 0.2 * np.sin(2 * math.pi * 55 * t)                       # mono, 48 kHz
teclado = np.stack([0.1 * np.sin(2 * math.pi * 440 * t),        # estéreo, 48 kHz
                    0.15 * np.sin(2 * math.pi * 440 * t + 0.3)], axis=1)
bat = np.zeros(sr * (seg - 5))                                    # mais curta
for i in range(0, len(bat) - 3000, sr // 2):
    bat[i:i + 3000] += rng.normal(0, 0.05, 3000) * np.exp(-np.arange(3000) / 500)
bat44 = signal.resample_poly(bat, 147, 160)                       # 44,1 kHz

with tempfile.TemporaryDirectory() as pasta:
    a_baixo = os.path.join(pasta, "Musica_Baixo.wav")
    a_tec = os.path.join(pasta, "Musica_Teclado.wav")
    a_bat = os.path.join(pasta, "Musica_Bateria.wav")
    sf.write(a_baixo, baixo, sr, subtype="PCM_24")
    sf.write(a_tec, teclado, sr, subtype="PCM_24")
    sf.write(a_bat, bat44, 44100, subtype="PCM_24")
    stems = [a_baixo, a_tec, a_bat]

    avisos = []
    soma, srs, lista = motor.carregar_stems(stems, {}, [], avisos)
    checar(srs == 48000 and soma.shape == (sr * seg, 2), f"soma: 48 kHz, estéreo, duração da maior ({soma.shape})")
    checar(any("44100" in a for a in avisos), "avisa e converte a stem de 44,1 kHz")
    checar(np.allclose(soma[:, 0] - soma[:, 1], teclado[:, 0] - teclado[:, 1], atol=1e-4),
           "stem mono vai igual nos dois lados")

    soma2, _, lista2 = motor.carregar_stems(stems, {a_baixo: -6.0}, [a_bat], [])
    checar(abs((db_banda(soma, sr, 40, 70) - db_banda(soma2, sr, 40, 70)) - 6) < 0.3, "volume -6 dB na stem do baixo")
    checar(db_banda(soma2, sr, 3000, 12000) < db_banda(soma, sr, 3000, 12000) - 30, "mudo tira a bateria")
    checar(lista2[2] == ("Musica_Bateria.wav", None), "lista marca a stem no mudo")

    alta = [os.path.join(pasta, f"Alta_{i}.wav") for i in range(4)]
    for c in alta:
        sf.write(c, np.stack([0.9 * np.sin(2 * math.pi * 200 * t)] * 2, axis=1), sr, subtype="FLOAT")
    soma3, _, _ = motor.carregar_stems(alta, {}, [], [])
    checar(np.max(np.abs(soma3)) <= 0.891, "soma alta é abaixada (sem passar de 0 dB)")

    try:
        motor.carregar_stems(stems, {}, stems, [])
        checar(False, "tudo no mudo deveria dar erro")
    except ValueError as e:
        checar("mudo" in str(e), "tudo no mudo dá erro claro")

    checar(motor.nome_base(stems) == os.path.join(pasta, "Musica"), "nome do master vem do começo comum dos nomes")
    checar(motor.nome_base([os.path.join(pasta, "a.wav"), os.path.join(pasta, "b.wav")]).endswith("Master das stems"),
           "sem nome comum: 'Master das stems'")

    r = motor.processar(stems, {"destino": "streaming", "alvo_lufs": -14, "stems_ganhos": {a_tec: 2.0},
                                "stems": {"voz": 3.0}})
    checar(abs(r["depois"]["lufs"] + 14) <= 0.5, f"master das stems: {r['depois']['lufs']:.2f} LUFS")
    checar(r["original"].shape == r["audio"].shape, "A/B: soma original e master do mesmo tamanho")
    checar(not any("aproximado" in f for f in r["feito"]), "rebalanceamento aproximado fica desligado com stems")
    checar(any("Soma de 3 stems" in f and "Musica_Teclado.wav +2.0 dB" in f for f in r["feito"]),
           "relatório lista as stems e os volumes")
    r = motor.salvar(r)
    checar(os.path.basename(r["saida"]) == "Musica_master.wav" and "Stems:   3 stems" in r["relatorio"],
           f"salvo como {os.path.basename(r['saida'])}")
    r1 = motor.processar([a_tec], {"destino": "streaming", "alvo_lufs": -14})
    checar(r1["stems"] is None, "lista com 1 arquivo = modo mix normal")

print()
print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
sys.exit(1 if falhas else 0)
