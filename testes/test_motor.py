"""Testes do masterizador com sinais sintéticos. Rode: python testes/test_motor.py"""

import math
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dsp  # noqa: E402
import motor  # noqa: E402
import stems  # noqa: E402
import player  # noqa: E402

falhas = 0


def checar(cond, texto):
    global falhas
    print(("ok   " if cond else "FALHA ") + texto)
    if not cond:
        falhas += 1


def mix_falsa(sr=48000, seg=40, canais=2, nivel=0.25, semente=1):
    """Acordes + 'bumbo' + 'caixa' de ruído, com parte fraca e parte forte."""
    rng = np.random.default_rng(semente)
    t = np.arange(int(sr * seg)) / sr
    x = np.zeros(len(t))
    for f in (110, 220, 277, 330, 440, 660):
        x += 0.15 * np.sin(2 * math.pi * f * t + rng.uniform(0, 6))
    batida = int(sr * 0.5)
    for i in range(0, len(t) - batida, batida):
        k = np.arange(int(sr * 0.15))
        x[i:i + len(k)] += 1.2 * np.sin(2 * math.pi * 55 * k / sr) * np.exp(-k / (sr * 0.04))
        j = i + batida // 2
        r = rng.normal(0, 1, len(k)) * np.exp(-k / (sr * 0.03))
        x[j:j + len(k)] += 0.6 * signal.lfilter(*signal.butter(2, 2000, "highpass", fs=sr), r)
    x *= np.where(t < seg / 2, 0.4, 1.0)  # verso fraco, refrão forte
    x = x / np.max(np.abs(x)) * nivel
    if canais == 2:
        x = np.stack([x, np.roll(x, 37) * 0.9], axis=1)
    else:
        x = x[:, None]
    return x


def energia(x):
    return float(np.mean(x ** 2))


def db_banda(x, sr, a, b):
    sos = signal.butter(4, [a, b], "bandpass", fs=sr, output="sos")
    return 10 * math.log10(energia(signal.sosfilt(sos, x, axis=0)) + 1e-30)


if __name__ != "__main__":  # importado por outro script (só quer mix_falsa)
    raise ImportError("só mix_falsa")

sr = 48000

# --- EQ ----------------------------------------------------------------------
sos = dsp.biquad(sr, "grave", 120, 3.0)
_, h = signal.sosfreqz(sos, worN=[30, 5000], fs=sr)
checar(abs(20 * np.log10(abs(h[0])) - 3) < 0.4 and abs(20 * np.log10(abs(h[1]))) < 0.1, "shelf de grave")
sos = dsp.biquad(sr, "agudo", 8000, -2.0)
_, h = signal.sosfreqz(sos, worN=[100, 18000], fs=sr)
checar(abs(20 * np.log10(abs(h[0]))) < 0.1 and abs(20 * np.log10(abs(h[1])) + 2) < 0.4, "shelf de agudo")
sos = dsp.biquad(sr, "sino", 3000, 4.0, 0.9)
_, h = signal.sosfreqz(sos, worN=[3000, 200], fs=sr)
checar(abs(20 * np.log10(abs(h[0])) - 4) < 0.1 and abs(20 * np.log10(abs(h[1]))) < 0.2, "sino de presença")

# --- loudness ------------------------------------------------------------------
x = mix_falsa()
import pyloudnorm as pyln  # noqa: E402
oficial = pyln.Meter(sr).integrated_loudness(x)
checar(abs(dsp.loudness(x, sr)["lufs"] - oficial) < 0.05, f"loudness rápido = pyloudnorm ({oficial:.2f})")

# --- multibanda: bandas somadas = original; e comprime ---------------------------
f1, f2 = dsp.MULTIBANDA_CORTES
s1 = signal.butter(2, f1, "lowpass", fs=sr, output="sos")
s2 = signal.butter(2, f2, "lowpass", fs=sr, output="sos")
g_ = dsp.por_canal(x, lambda c: dsp.filtrar(s1, c, True))
m_ = dsp.por_canal(x - g_, lambda c: dsp.filtrar(s2, c, True))
a_ = x - g_ - m_
checar(np.max(np.abs(g_ + m_ + a_ - x)) < 1e-12, "multibanda: bandas somadas = original")
y, info = dsp.multibanda(x, sr, "media")
checar(len(info) == 3 and all(2.5 < i["gr"] < 3.5 for i in info), f"multibanda média: {[round(i['gr'], 1) for i in info]} dB")
y, info = dsp.multibanda(x, sr, "nenhuma")
checar(y is x and info == [], "multibanda desligada não mexe")

# --- EQ dinâmico: abaixa a aspereza só quando aparece ---------------------------
t = np.arange(sr * 20) / sr
base = 0.2 * np.sin(2 * math.pi * 220 * t) + 0.05 * np.sin(2 * math.pi * 3500 * t)
aspero = 0.25 * np.sin(2 * math.pi * 3500 * t) * ((t % 2) < 0.5)  # rajadas de 3,5 kHz
xd = np.stack([base + aspero] * 2, axis=1)
yd, info = dsp.eq_dinamico(xd, sr, ["aspereza"], 6.0)
checar(len(info) == 1 and 3000 < info[0]["freq"] < 4200, f"EQ dinâmico achou a aspereza em {info[0]['freq']:.0f} Hz")
rajada = (t % 2) < 0.5
calmo = ((t % 2) > 1.0) & ((t % 2) < 1.9)
red_rajada = db_banda(xd[rajada], sr, 3000, 4000) - db_banda(yd[rajada], sr, 3000, 4000)
red_calmo = db_banda(xd[calmo], sr, 3000, 4000) - db_banda(yd[calmo], sr, 3000, 4000)
checar(red_rajada > 2.0 and red_calmo < 0.5, f"EQ dinâmico: -{red_rajada:.1f} dB na rajada, -{red_calmo:.1f} dB no calmo")
checar(abs(db_banda(xd, sr, 150, 300) - db_banda(yd, sr, 150, 300)) < 0.2, "EQ dinâmico não mexe no resto")

# --- match EQ: aproxima o espectro da referência --------------------------------
rng = np.random.default_rng(3)
ruido = rng.normal(0, 0.1, (sr * 20, 2))
escuro = dsp.por_canal(ruido, lambda c: signal.sosfilt(dsp.biquad(sr, "agudo", 4000, -6.0), c))
fc = dsp.centros_terco(sr)
cm, cr = dsp.curva_terco(escuro, sr), dsp.curva_terco(ruido, sr)
curva = dsp.curva_match(cm, cr, fc, 1.0)
checar(curva[fc >= 8000].mean() > 3 and abs(curva[(fc > 150) & (fc < 1000)].mean()) < 1.5,
       f"match EQ: agudos {curva[fc >= 8000].mean():+.1f} dB, médios {curva[(fc > 150) & (fc < 1000)].mean():+.1f} dB")
ym = dsp.aplicar_curva(escuro, sr, fc, curva)
faixa = (fc > 100) & (fc < 12000)


def dif_forma(a, b):  # diferença de formato (sem contar o volume geral)
    d = (a - b)[faixa]
    return float(np.mean(np.abs(d - np.mean(d))))


antes_d = dif_forma(cm, cr)
depois_d = dif_forma(dsp.curva_terco(ym, sr), cr)
checar(depois_d < antes_d * 0.5, f"match EQ aproxima da referência (dif. {antes_d:.1f} → {depois_d:.1f} dB)")
checar(np.max(np.abs(dsp.curva_match(cm - 30, cr, fc, 1.0))) <= 6.0, "match EQ respeita ±6 dB")

# --- mid/side e estéreo ----------------------------------------------------------
lado = ruido
ys = dsp.imagem_estereo(lado, sr, 120, 1.0)
m_ant, s_antes = dsp.para_mid_side(lado)
m_dep, s_dep = dsp.para_mid_side(ys)
checar(db_banda(s_dep[:, None], sr, 30, 60) < db_banda(s_antes[:, None], sr, 30, 60) - 15, "grave em mono: side some no grave")
checar(np.allclose(m_dep, m_ant), "estéreo não mexe no centro")
yl = dsp.imagem_estereo(lado, sr, None, 1.5)
_, s_l = dsp.para_mid_side(yl)
checar(abs(db_banda(s_l[:, None], sr, 2000, 8000) - db_banda(s_antes[:, None], sr, 2000, 8000) - 3.5) < 0.3,
       "largura 150% abre ~3,5 dB nas laterais")
yms = dsp.eq_mid_side(lado, sr, {"grave": 0, "presenca": 4.0, "agudo": 0}, {"grave": 0, "agudo": -3.0})
mm, ss = dsp.para_mid_side(yms)
checar(abs(db_banda(mm[:, None], sr, 2800, 3200) - db_banda(m_ant[:, None], sr, 2800, 3200) - 4) < 0.6
       and abs(db_banda(ss[:, None], sr, 12000, 16000) - db_banda(s_antes[:, None], sr, 12000, 16000) + 3) < 0.6,
       "EQ mid/side: +4 no centro em 3 kHz, -3 nas laterais em 12 kHz")
ymono = dsp.eq_mid_side(ruido[:, :1], sr, {"grave": 2, "presenca": 0, "agudo": 0}, {"grave": 0, "agudo": 0})
checar(ymono.shape[1] == 1, "EQ mid/side em arquivo mono")

# --- soft clip ---------------------------------------------------------------------
yc = dsp.soft_clip(x, 2.0)
reducao = dsp.db_amp(np.max(np.abs(x))) - dsp.db_amp(np.max(np.abs(yc)))
checar(1.5 < reducao < 2.6, f"soft clip abaixa os picos em {reducao:.1f} dB")
checar(abs(db_banda(x, sr, 200, 2000) - db_banda(yc, sr, 200, 2000)) < 0.5, "soft clip quase não muda o corpo")

# --- stems -------------------------------------------------------------------------
tt = np.arange(sr * 12) / sr
tonal = 0.2 * np.sin(2 * math.pi * 440 * tt) + 0.2 * np.sin(2 * math.pi * 660 * tt)
baixo = 0.3 * np.sin(2 * math.pi * 60 * tt)
bat = np.zeros_like(tt)
for i in range(0, len(tt) - 2000, sr // 4):
    bat[i:i + 1500] += rng.normal(0, 0.5, 1500) * np.exp(-np.arange(1500) / 300)
xs = np.stack([tonal + baixo + bat] * 2, axis=1)
ys, g = stems.rebalancear(xs, sr, {"bateria": -6})
d_bat = db_banda(xs, sr, 5000, 12000) - db_banda(ys, sr, 5000, 12000)
d_ton = db_banda(xs, sr, 420, 460) - db_banda(ys, sr, 420, 460)
checar(d_bat > 3 and d_ton < 1.5, f"stems: bateria -{d_bat:.1f} dB, tonal -{d_ton:.1f} dB")
ys, g = stems.rebalancear(xs, sr, {"baixo": 4})
d_b = db_banda(ys, sr, 50, 70) - db_banda(xs, sr, 50, 70)
d_t = db_banda(ys, sr, 420, 460) - db_banda(xs, sr, 420, 460)
checar(d_b > 2.5 and abs(d_t) < 1.0, f"stems: baixo +{d_b:.1f} dB, tonal {d_t:+.1f} dB")
ys, g = stems.rebalancear(xs, sr, {"voz": 4})
d_v = db_banda(ys, sr, 420, 460) - db_banda(xs, sr, 420, 460)
checar(d_v > 2.5, f"stems: voz (centro sustentado) +{d_v:.1f} dB")
# duração "quebrada" (vários trechos, o último curto): sem erro e sem emenda audível
xl = np.concatenate([xs] * 4)[: int(sr * 40.3)]
yl_, _ = stems.rebalancear(xl, sr, {"bateria": -3})
checar(yl_.shape == xl.shape, "stems: áudio longo com último trecho curto")
yz_, _ = stems.rebalancear(xl, sr, {"bateria": 0.1})
d = np.abs(yz_ - xl).max()
checar(d < 0.05, f"stems: emendas entre trechos não aparecem (dif. máx. {d:.3f})")
ys, g = stems.rebalancear(xs, sr, {})
checar(ys is xs, "stems neutros não mexem")
lados = np.stack([tonal + bat, -tonal + bat], axis=1)  # tonal fora do centro
ys, _ = stems.rebalancear(lados, sr, {"voz": 6})
checar(abs(db_banda(ys, sr, 420, 460) - db_banda(lados, sr, 420, 460)) < 1.5, "stems: voz não pega o que está nos lados")

# --- player (sem placa de som: chama o callback direto) ----------------------------
pl = player.Player()
orig = np.ones((1000, 1)) * 0.5
mast = np.ones((1000, 2)) * 0.8
pl.carregar(orig, mast, 1000, -20.0, -14.0)
saida = np.zeros((300, 2), dtype=np.float32)


class Parou(Exception):
    pass


if player.sd is None:
    player.sd = type("sd", (), {"CallbackStop": Parou})
pl.escolher("b")
pl._callback(saida, 300, None, None)
checar(abs(saida[0, 0] - 0.8 * 10 ** (-6 / 20)) < 1e-4, "player: master abaixado 6 dB para igualar")
pl.escolher("a")
pl._callback(saida, 300, None, None)
checar(abs(saida[0, 1] - 0.5) < 1e-6 and pl.pos == 600, "player: A/B continua do mesmo ponto")
pl.definir_igualar(False)
pl.escolher("b")
saida = np.zeros((600, 2), dtype=np.float32)
try:
    pl._callback(saida, 600, None, None)
except Exception:
    pass
checar(abs(saida[0, 0] - 0.8) < 1e-6 and saida[-1, 0] == 0 and pl.pos == 1000, "player: fim do arquivo com silêncio")

# --- cadeia completa ----------------------------------------------------------------
with tempfile.TemporaryDirectory() as pasta:
    tudo = {"dinamico": "medio", "dinamico_faixas": ["embolado", "aspereza", "sibilancia"],
            "multibanda": "leve", "compressao": "leve", "ms_mid": {"presenca": 1.0}, "ms_side": {"agudo": 1.0},
            "largura": 115, "stems": {"voz": 1.5, "bateria": -1.0, "baixo": 1.0}, "saturacao": True,
            "limitador": "softclip"}
    casos = [
        ("streaming leve", dict(destino="streaming", alvo_lufs=-14), {}),
        ("show pegada", dict(destino="show", alvo_lufs=-10, compressao="media", limitador="pegada"), {}),
        ("mono 44.1k sem comp", dict(destino="personalizado", alvo_lufs=-12, compressao="nenhuma"), dict(sr=44100, canais=1)),
        ("teto -2 e EQ manual", dict(destino="personalizado", alvo_lufs=-13, teto_tp=-2.0, grave=2.5, agudo=-1.5), {}),
        ("tudo ligado", dict(destino="show", alvo_lufs=-10, **tudo), {}),
        ("tudo ligado mono", dict(destino="streaming", alvo_lufs=-14, **tudo), dict(canais=1)),
    ]
    for nome, cfg, gerar in casos:
        srx = gerar.get("sr", 48000)
        xx = mix_falsa(**gerar)
        arq = os.path.join(pasta, nome.replace(" ", "_") + ".wav")
        sf.write(arq, xx, srx, subtype="PCM_24")
        t0 = time.time()
        r = motor.salvar(motor.processar(arq, cfg))
        yy, sry = sf.read(r["saida"], always_2d=True)
        info = sf.info(r["saida"])
        lufs = dsp.loudness(yy, sry)["lufs"]
        tp = dsp.true_peak(yy)
        checar(sry == srx and yy.shape == xx.shape and info.subtype == "PCM_24", f"{nome}: formato mantido")
        checar(abs(lufs - cfg["alvo_lufs"]) <= 0.5, f"{nome}: {lufs:.2f} LUFS (alvo {cfg['alvo_lufs']})")
        checar(tp <= cfg.get("teto_tp", -1.0) + 0.1, f"{nome}: true peak {tp:.2f}  [{time.time() - t0:.1f} s]")
        checar(os.path.exists(r["arquivo_relatorio"]) and "ainda não salvo" not in r["relatorio"], f"{nome}: relatório")
    checar(any("Multibanda" in f for f in r["feito"]) and any("Rebalanceamento" in f for f in r["feito"])
           and any("Soft clip" in f for f in r["feito"]), "relatório lista as etapas ligadas")

    # processar não grava nada
    arq = os.path.join(pasta, "so_ouvir.wav")
    sf.write(arq, mix_falsa(seg=10), 48000, subtype="PCM_24")
    r = motor.processar(arq, {"destino": "streaming", "alvo_lufs": -14})
    checar(not any(f.startswith("so_ouvir_master") for f in os.listdir(pasta)), "processar (ouvir) não grava arquivo")
    checar(r["original"].shape == r["audio"].shape, "original e master do mesmo tamanho (para o A/B)")

    # referência com match EQ
    ref = mix_falsa(semente=7)
    ref = dsp.por_canal(ref, lambda c: signal.sosfilt(dsp.biquad(48000, "agudo", 6000, 4.0), c))
    ref_arq = os.path.join(pasta, "ref.wav")
    yref = dsp.ajustar_loudness(ref, 48000, -9.0, -1.0)["audio"]
    sf.write(ref_arq, yref, 48000, subtype="PCM_24")
    arq = os.path.join(pasta, "mix_ref.wav")
    sf.write(arq, mix_falsa(), 48000, subtype="PCM_24")
    r = motor.masterizar(arq, {"destino": "referencia", "referencia": ref_arq, "match_eq": 80})
    checar(abs(r["depois"]["lufs"] - r["referencia"]["lufs"]) <= 0.5,
           f"referência: {r['depois']['lufs']:.2f} vs ref {r['referencia']['lufs']:.2f}")
    checar(any("Match EQ" in f for f in r["feito"]) and r["grave"] == 0 and r["agudo"] == 0,
           "com match EQ, o tilt automático fica em 0")

    r2 = motor.masterizar(arq, {"destino": "streaming", "alvo_lufs": -14})
    checar(r2["saida"].endswith("_master_2.wav"), "segundo master ganha outro nome")

    xc = np.clip(mix_falsa(nivel=4.0), -1, 1)
    arq = os.path.join(pasta, "clip.wav")
    sf.write(arq, xc, 48000, subtype="FLOAT")
    r = motor.processar(arq, {"destino": "streaming", "alvo_lufs": -14})
    checar(any("clipada" in a for a in r["avisos"]), "aviso de mix clipada")

    xz = np.concatenate([np.zeros((48000 * 60, 2)), mix_falsa(seg=10), np.zeros((48000 * 60, 2))])
    t0 = time.time()
    dsp.loudness(xz, 48000)
    dsp.equalizar(xz, 48000, 1.0, 1.0)
    dsp.multibanda(xz, 48000, "leve")
    dt = time.time() - t0
    checar(dt < 30, f"silêncio digital não trava os filtros ({dt:.1f} s)")

    arq = os.path.join(pasta, "mudo.wav")
    sf.write(arq, np.zeros((48000 * 3, 2)), 48000)
    try:
        motor.processar(arq, {"destino": "streaming", "alvo_lufs": -14})
        checar(False, "silêncio deveria dar erro")
    except ValueError as e:
        checar("silêncio" in str(e), "silêncio dá erro claro")

print()
print("TUDO OK" if falhas == 0 else f"{falhas} FALHA(S)")
sys.exit(1 if falhas else 0)
