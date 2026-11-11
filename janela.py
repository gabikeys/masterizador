"""
janela.py — interface do masterizador, no estilo de plugin de estúdio.

Cadeia de módulos (DESTINO → EQ → DINÂMICA → ESTÉREO → STEMS → LIMITADOR),
analisador de espectro, medidores (pico, LUFS momentâneo, redução do limitador)
e transporte com forma de onda e A/B. Tudo desenhado com Tkinter (já vem com o
Python do python.org), sem instalar nada além das bibliotecas de áudio.

Fluxo: ABRIR (mix ou stems) → ajustar → PROCESSAR → ouvir A/B → SALVAR.
As escolhas ficam salvas em ~/.masterizador_config.json.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA)

try:
    import numpy as np
    import dsp
    import motor
    from player import Player
    import ui_widgets as ui
    from ui_widgets import (FUNDO, PAINEL, PAINEL_TOPO, BORDA, TEXTO, TEXTO_FRACO, TEXTO_APAGADO, DESTAQUE,
                            VERDE, VERMELHO, AMARELO, AZUL, TRILHO, fonte, Knob, Botao, Segmentado, Seletor,
                            Painel, Fader, rotulo)
except ImportError as erro:  # bibliotecas não instaladas
    if "--autoteste" in sys.argv:
        print(f"AUTOTESTE FALHOU: faltou biblioteca: {erro}")
        sys.exit(1)
    raiz = tk.Tk()
    raiz.withdraw()
    messagebox.showerror(
        "Masterizador",
        "Faltam bibliotecas de áudio.\n\nDê dois cliques em 'instalar.command' (Mac) ou rode no Terminal:\n\n"
        "python3 -m pip install numpy scipy soundfile pyloudnorm sounddevice"
        f"\n\n(detalhe: {erro})")
    sys.exit(1)

ARQ_CONFIG = os.path.join(os.path.expanduser("~"), ".masterizador_config.json")
TIPOS_AUDIO = [("Áudio", "*.wav *.WAV *.aif *.aiff *.AIF *.AIFF *.flac *.FLAC *.mp3 *.MP3"), ("Todos", "*.*")]
TIPOS_REF = [("Áudio", "*.wav *.WAV *.aif *.aiff *.AIF *.AIFF *.flac *.FLAC *.mp3 *.MP3"), ("Todos", "*.*")]
CHAVES_CONFIG = ("destino", "alvo_lufs", "teto_tp", "estilo", "referencia", "grave", "agudo", "match_eq",
                 "ms_mid", "ms_side", "dinamico", "dinamico_faixas", "multibanda", "compressao", "mono_grave",
                 "mono_grave_hz", "largura", "stems", "saturacao", "limitador")
ALTURA_MODULOS = 292


def ler_config() -> dict:
    c = motor.completar({})
    c.update({"pasta": os.path.expanduser("~"), "grave_valor": 0.0, "agudo_valor": 0.0,
              "on_eq": True, "on_din": True, "on_est": True, "on_stems": True})
    try:
        with open(ARQ_CONFIG, encoding="utf-8") as f:
            salvo = json.load(f)
        c = {**c, **motor.completar({k: v for k, v in salvo.items() if k in CHAVES_CONFIG}),
             **{k: v for k, v in salvo.items() if k not in CHAVES_CONFIG}}
    except Exception:
        pass
    return c


def gravar_config(c: dict) -> None:
    try:
        with open(ARQ_CONFIG, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def abrir_na_pasta(caminho: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", caminho])
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", os.path.normpath(caminho)])
        else:
            subprocess.run(["xdg-open", os.path.dirname(caminho)])
    except Exception:
        pass


def mmss(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


def quadro(pai, **kw):
    return tk.Frame(pai, bg=pai.cget("bg"), **kw)


class App:
    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.cfg = ler_config()
        self.fila: queue.Queue = queue.Queue()
        self.trabalhando = False
        self.resultado = None
        self.cfg_processada = None
        self.player = Player()
        self.arquivos: list[str] = []
        self.stem_ganhos: dict[str, float] = {}
        self.stem_mudos: set[str] = set()
        self.ultima_saida = None
        self.progresso = 0.0
        self.mixer = None
        self.janela_rel = None
        self.medidor = {"L": -90.0, "R": -90.0, "M": -60.0, "GR": 0.0, "holdL": -90.0, "holdR": -90.0, "t_hold": 0}
        self.esp_vivo = None
        self._vivo_visivel = False
        self.onda = None
        self.arrastando_onda = False
        self.montando = True

        raiz.title("Masterizador")
        raiz.configure(bg=FUNDO)
        raiz.geometry("1210x760")
        raiz.minsize(1150, 720)

        self.cabecalho()
        self.linha_modulos()
        self.linha_analise()
        self.linha_transporte()

        self.montando = False
        self.mudou_destino(inicial=True)
        self.atualizar()
        for tecla, f in (("<space>", self.tocar_parar), ("<Key-a>", lambda: self.lado("a")),
                         ("<Key-b>", lambda: self.lado("b")), ("<Left>", lambda: self.pular(-5)),
                         ("<Right>", lambda: self.pular(5))):
            raiz.bind(tecla, lambda e, f=f: (f(), "break")[1])
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)
        raiz.after(50, self.ciclo)

    # =================================================================== montagem
    def cabecalho(self):
        h = quadro(self.raiz, height=56)
        h.pack(fill="x", padx=10, pady=(8, 4))
        h.pack_propagate(False)
        logo = tk.Canvas(h, width=210, height=50, bg=FUNDO, highlightthickness=0)
        logo.pack(side="left")
        for i, a in enumerate((6, 14, 22, 30, 20, 12, 26, 34, 18, 8)):
            logo.create_line(6 + i * 4, 25 - a / 2, 6 + i * 4, 25 + a / 2, fill=DESTAQUE, width=2)
        logo.create_text(52, 20, text="MASTERIZADOR", anchor="w", fill=TEXTO, font=fonte(13, "bold"))
        logo.create_text(53, 37, text="mastering · stems · A/B", anchor="w", fill=TEXTO_FRACO, font=fonte(7.5))
        Botao(h, "ABRIR MIX / STEMS", 150, 30, comando=self.escolher_arquivos, tam_fonte=8,
              dica="1 arquivo = mix pronta. Vários de uma vez (segure ⌘/Ctrl ou Shift) = stems.").pack(
            side="left", padx=(6, 10))
        caixa = tk.Frame(h, bg=TRILHO, highlightbackground=BORDA, highlightthickness=1)
        caixa.pack(side="left", fill="x", expand=True, ipady=4)
        self.l_arquivo = tk.Label(caixa, text="Nenhum arquivo — clique em ABRIR MIX / STEMS", bg=TRILHO,
                                  fg=TEXTO_FRACO, font=fonte(8.5), anchor="w")
        self.l_arquivo.pack(fill="x", padx=8)
        direita = quadro(h)
        direita.pack(side="right", padx=(10, 0))
        self.l_status = tk.Label(direita, text="", bg=FUNDO, fg=TEXTO_FRACO, font=fonte(8), anchor="e", width=62)
        self.l_status.pack(anchor="e")
        self.c_prog = tk.Canvas(direita, width=300, height=6, bg=TRILHO, highlightthickness=0)
        self.c_prog.pack(anchor="e", pady=(4, 0))

    def linha_modulos(self):
        linha = quadro(self.raiz, height=ALTURA_MODULOS)
        linha.pack(fill="x", padx=10, pady=4)
        c = self.cfg
        m = self.mudou

        # ---------- DESTINO
        p = Painel(linha, "DESTINO", 222, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y", padx=(0, 6))
        b = p.corpo
        self.seg_destino = Segmentado(b, [("streaming", "STREAM  -14"), ("show", "SHOW  -10"),
                                          ("referencia", "REFERÊNCIA"), ("personalizado", "CUSTOM")],
                                      c["destino"], comando=lambda k: self.mudou_destino(), largura=102, colunas=2)
        self.seg_destino.pack(pady=(8, 2))
        k = quadro(b)
        k.pack()
        self.k_alvo = Knob(k, "ALVO LUFS", motor.LUFS_MIN, motor.LUFS_MAX, c["alvo_lufs"], 0.5, padrao=-14.0,
                           formato="{:.1f}", bipolar=False, ao_mudar=lambda v: self.alvo_mexido(),
                           dica="Loudness integrado final. -14 = streaming, -10 = show.")
        self.k_alvo.pack(side="left", padx=4)
        self.k_teto = Knob(k, "TETO dBTP", motor.TETO_MIN, motor.TETO_MAX, c["teto_tp"], 0.1, padrao=-1.0,
                           formato="{:.1f}", bipolar=False, ao_mudar=lambda v: m(),
                           dica="Pico máximo (true peak). -1,0 é o seguro para streaming.")
        self.k_teto.pack(side="left", padx=4)
        self.v_ref = c.get("referencia") or ""
        self.b_ref = Botao(b, self.texto_ref(), 204, 24, comando=self.escolher_ref, tam_fonte=7.5,
                           dica="Música de referência (WAV, AIFF ou MP3): copia o loudness e a curva dela.")
        self.b_ref.pack(pady=(4, 6))
        rotulo(b, "ESTILO (EQ AUTOMÁTICO)").pack()
        self.sel_estilo = Seletor(b, [(k_, v[0].upper()) for k_, v in motor.ESTILOS.items()], c["estilo"], 204,
                                  comando=lambda k_: m())
        self.sel_estilo.pack(pady=(2, 0))

        # ---------- EQ
        p = Painel(linha, "EQUALIZAÇÃO", 288, com_liga=True, ligado=c["on_eq"], ao_ligar=m, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y", padx=(0, 6))
        self.p_eq = p
        b = p.corpo
        k = quadro(b)
        k.pack(pady=(6, 0))
        col = quadro(k)
        col.pack(side="left", padx=3)
        self.k_grave = Knob(col, "GRAVES", -3, 3, c.get("grave_valor", 0.0), 0.5, ao_mudar=lambda v: m(),
                            dica="Shelf em 120 Hz (±3 dB).")
        self.k_grave.pack()
        self.b_grave_auto = Botao(col, "AUTO", 60, 20, modo="liga", ligado=c["grave"] is None, comando=m, tam_fonte=7,
                                  dica="Automático: compara com o estilo (ou a referência) e corrige até ±2 dB.")
        self.b_grave_auto.pack()
        col = quadro(k)
        col.pack(side="left", padx=3)
        self.k_agudo = Knob(col, "AGUDOS", -3, 3, c.get("agudo_valor", 0.0), 0.5, ao_mudar=lambda v: m(),
                            dica="Shelf em 8 kHz (±3 dB).")
        self.k_agudo.pack()
        self.b_agudo_auto = Botao(col, "AUTO", 60, 20, modo="liga", ligado=c["agudo"] is None, comando=m, tam_fonte=7)
        self.b_agudo_auto.pack()
        col = quadro(k)
        col.pack(side="left", padx=3)
        self.k_match = Knob(col, "MATCH EQ", 0, 100, c["match_eq"], 10, padrao=60, formato="{:.0f}", unidade="%",
                            bipolar=False, ao_mudar=lambda v: m(),
                            dica="Quanto copiar da curva de frequências da referência (só com REFERÊNCIA).")
        self.k_match.pack()
        rotulo(col, "só c/ referência", 6.5, peso="normal").pack()
        tk.Frame(b, bg=BORDA, height=1).pack(fill="x", padx=10, pady=(8, 2))
        rotulo(b, "MID  (centro: voz, bumbo, baixo)          SIDE  (laterais)", 6.5).pack()
        k = quadro(b)
        k.pack()
        mid, side = c["ms_mid"], c["ms_side"]
        self.k_ms = {}
        for chave, rot, val, dica in (("mid_grave", "M GRAVE", mid["grave"], "Centro, shelf 120 Hz"),
                                      ("mid_presenca", "M PRES", mid["presenca"], "Centro, 3 kHz: destaca a voz"),
                                      ("mid_agudo", "M AGUDO", mid["agudo"], "Centro, shelf 8 kHz"),
                                      ("side_grave", "S GRAVE", side["grave"], "Laterais, shelf 200 Hz"),
                                      ("side_agudo", "S AGUDO", side["agudo"], "Laterais, shelf 8 kHz: abre o som")):
            self.k_ms[chave] = Knob(k, rot, -4, 4, val, 0.5, tam=32, ao_mudar=lambda v: m(), dica=dica)
            self.k_ms[chave].pack(side="left", padx=0)

        # ---------- DINÂMICA
        p = Painel(linha, "DINÂMICA", 224, com_liga=True, ligado=c["on_din"], ao_ligar=m, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y", padx=(0, 6))
        self.p_din = p
        b = p.corpo
        rotulo(b, "EQ DINÂMICO").pack(pady=(8, 2))
        self.seg_din = Segmentado(b, [("desligado", "OFF"), ("leve", "LEVE"), ("medio", "MÉDIO")], c["dinamico"],
                                  comando=lambda k_: m(), largura=66)
        self.seg_din.pack()
        k = quadro(b)
        k.pack(pady=(3, 0))
        self.b_faixas = {}
        for chave, rot, dica in (("embolado", "EMB", "Embolado (180–500 Hz)"), ("aspereza", "ASP", "Aspereza (2–5 kHz)"),
                                 ("sibilancia", "SIB", "Sibilância (5–10 kHz)")):
            self.b_faixas[chave] = Botao(k, rot, 66, 22, modo="liga", ligado=chave in c["dinamico_faixas"],
                                         comando=m, tam_fonte=7, dica=dica + ": acha a frequência e só abaixa quando passa do normal.")
            self.b_faixas[chave].pack(side="left", padx=1)
        rotulo(b, "MULTIBANDA  (150 Hz · 2,5 kHz)").pack(pady=(12, 2))
        self.seg_multi = Segmentado(b, [("nenhuma", "OFF"), ("leve", "LEVE"), ("media", "MÉDIA")], c["multibanda"],
                                    comando=lambda k_: m(), largura=66)
        self.seg_multi.pack()
        rotulo(b, "COMPRESSOR DE COLA").pack(pady=(12, 2))
        self.seg_cola = Segmentado(b, [("nenhuma", "OFF"), ("leve", "LEVE"), ("media", "MÉDIA")], c["compressao"],
                                   comando=lambda k_: m(), largura=66)
        self.seg_cola.pack()

        # ---------- ESTÉREO
        p = Painel(linha, "ESTÉREO", 136, com_liga=True, ligado=c["on_est"], ao_ligar=m, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y", padx=(0, 6))
        self.p_est = p
        b = p.corpo
        self.b_mono = Botao(b, "MONO BASS", 116, 24, modo="liga", ligado=c["mono_grave"], comando=m, tam_fonte=7.5,
                            dica="Grave em mono abaixo da frequência escolhida: mais firme no show, carro e celular.")
        self.b_mono.pack(pady=(8, 2))
        self.k_mono_hz = Knob(b, "ABAIXO DE", 60, 200, float(c["mono_grave_hz"]), 10, padrao=120, formato="{:.0f}",
                              unidade=" Hz", bipolar=False, ao_mudar=lambda v: m())
        self.k_mono_hz.pack()
        self.k_largura = Knob(b, "LARGURA", 50, 150, float(c["largura"]), 5, padrao=100, formato="{:.0f}",
                              unidade="%", centro=100, ao_mudar=lambda v: m(),
                              dica="100% = como está. Acima de ~130% pode perder coisas em mono.")
        self.k_largura.pack(pady=(4, 0))

        # ---------- STEMS
        p = Painel(linha, "STEMS", 140, com_liga=True, ligado=c["on_stems"], ao_ligar=m, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y", padx=(0, 6))
        self.p_stems = p
        b = p.corpo
        self.q_aprox = quadro(b)
        self.k_stems = {}
        for chave, rot in (("voz", "VOZ"), ("bateria", "BATERIA"), ("baixo", "BAIXO")):
            self.k_stems[chave] = Knob(self.q_aprox, rot, -6, 6, c["stems"].get(chave, 0.0), 0.5, tam=36,
                                       ao_mudar=lambda v: m(),
                                       dica="Aproximado: separa pelo tipo de som dentro da mix. Use ±2 a 3 dB.")
            self.k_stems[chave].pack(pady=0)
        rotulo(self.q_aprox, "aproximado (mix pronta)", 6.5, peso="normal").pack()
        self.q_lista = quadro(b)
        self.l_qtd = tk.Label(self.q_lista, text="", bg=PAINEL, fg=DESTAQUE, font=fonte(20, "bold"))
        self.l_qtd.pack(pady=(30, 0))
        rotulo(self.q_lista, "STEMS CARREGADAS").pack()
        Botao(self.q_lista, "MIXER", 110, 30, comando=self.abrir_mixer, destaque=True, tam_fonte=9,
              dica="Volume e mudo de cada stem (exato).").pack(pady=(16, 8))
        self.l_resumo_stems = tk.Label(self.q_lista, text="", bg=PAINEL, fg=TEXTO_FRACO, font=fonte(7.5),
                                       wraplength=120, justify="center")
        self.l_resumo_stems.pack()
        self.q_aprox.pack(fill="both", expand=True, pady=(4, 0))

        # ---------- LIMITADOR
        p = Painel(linha, "LIMITADOR", 140, height=ALTURA_MODULOS)
        p.pack(side="left", fill="y")
        b = p.corpo
        rotulo(b, "MODO").pack(pady=(8, 2))
        self.seg_lim = Segmentado(b, [("transparente", "TRANSPARENTE"), ("pegada", "PEGADA"), ("softclip", "SOFT CLIP")],
                                  c["limitador"], comando=lambda k_: m(), largura=120, altura=28, vertical=True)
        self.seg_lim.pack()
        self.b_sat = Botao(b, "SATURAÇÃO", 120, 26, modo="liga", ligado=c["saturacao"], comando=m, tam_fonte=7.5,
                           dica="Saturação leve tipo fita: calor e 'cola'.")
        self.b_sat.pack(pady=(14, 0))
        self.l_lim_info = tk.Label(b, text="", bg=PAINEL, fg=TEXTO_FRACO, font=fonte(7), wraplength=124, justify="center")
        self.l_lim_info.pack(pady=(10, 0))

    def linha_analise(self):
        linha = quadro(self.raiz)
        linha.pack(fill="both", expand=True, padx=10, pady=4)
        # espectro
        caixa = tk.Frame(linha, bg=BORDA)
        caixa.pack(side="left", fill="both", expand=True, padx=(0, 6))
        topo = tk.Frame(caixa, bg=PAINEL_TOPO, height=24)
        topo.pack(fill="x", padx=1, pady=(1, 0))
        topo.pack_propagate(False)
        tk.Label(topo, text="ANALISADOR", bg=PAINEL_TOPO, fg=TEXTO, font=fonte(8.5, "bold")).pack(side="left", padx=8)
        self.l_legenda = tk.Canvas(topo, width=330, height=22, bg=PAINEL_TOPO, highlightthickness=0)
        self.l_legenda.pack(side="right", padx=6)
        x = 4
        for cor, txt, tracejado in ((AZUL, "antes", False), (DESTAQUE, "depois", False), (TEXTO, "referência", True),
                                    ("#6b4a1c", "ao vivo", False)):
            self.l_legenda.create_line(x, 11, x + 16, 11, fill=cor, width=3 if txt == "ao vivo" else 2,
                                       dash=(3, 2) if tracejado else None)
            self.l_legenda.create_text(x + 20, 11, text=txt, anchor="w", fill=TEXTO_FRACO, font=fonte(7.5))
            x += 80
        self.c_esp = tk.Canvas(caixa, bg="#101214", highlightthickness=0)
        self.c_esp.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.c_esp.bind("<Configure>", lambda e: self.desenhar_espectro())
        # medidores
        caixa = tk.Frame(linha, bg=BORDA, width=360)
        caixa.pack(side="left", fill="y")
        caixa.pack_propagate(False)
        topo = tk.Frame(caixa, bg=PAINEL_TOPO, height=24)
        topo.pack(fill="x", padx=1, pady=(1, 0))
        topo.pack_propagate(False)
        tk.Label(topo, text="MEDIDORES", bg=PAINEL_TOPO, fg=TEXTO, font=fonte(8.5, "bold")).pack(side="left", padx=8)
        self.c_med = tk.Canvas(caixa, bg=PAINEL, highlightthickness=0)
        self.c_med.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.c_med.bind("<Configure>", lambda e: self.desenhar_medidores())

    def linha_transporte(self):
        linha = tk.Frame(self.raiz, bg=PAINEL, highlightbackground=BORDA, highlightthickness=1, height=96)
        linha.pack(fill="x", padx=10, pady=(4, 10))
        linha.pack_propagate(False)
        esq = tk.Frame(linha, bg=PAINEL)
        esq.pack(side="left", padx=10)
        self.b_play = Botao(esq, "▶  PLAY", 92, 40, comando=self.tocar_parar, tam_fonte=10,
                            dica="Tocar / parar (barra de espaço)")
        self.b_play.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self.seg_ab = Segmentado(esq, [("a", "A · ORIGINAL"), ("b", "B · MASTER")], "b",
                                 comando=lambda k: self.lado(k), largura=104, altura=40, tam_fonte=8.5)
        self.seg_ab.grid(row=0, column=1, rowspan=2)
        self.b_igualar = Botao(esq, "IGUALAR VOLUME", 132, 22, modo="liga", ligado=True, tam_fonte=7,
                               comando=lambda: self.player.definir_igualar(self.b_igualar.get()),
                               dica="Toca A e B no mesmo loudness: compare o SOM, não o volume.")
        self.b_igualar.grid(row=0, column=2, padx=(8, 0), sticky="s")
        self.l_teclas = tk.Label(esq, text="espaço · A/B · ← →", bg=PAINEL, fg=TEXTO_APAGADO, font=fonte(7))
        self.l_teclas.grid(row=1, column=2, padx=(8, 0), sticky="n")
        # direita
        dir_ = tk.Frame(linha, bg=PAINEL)
        dir_.pack(side="right", padx=10)
        self.b_processar = Botao(dir_, "PROCESSAR", 132, 40, comando=self.iniciar, destaque=True, tam_fonte=10,
                                 dica="Roda a master com os ajustes atuais (não salva nada ainda).")
        self.b_processar.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self.b_salvar = Botao(dir_, "SALVAR", 104, 40, comando=self.salvar, destaque=True, cor=VERDE, tam_fonte=10,
                              dica="Grava o master (WAV 24 bits) e o relatório na pasta da mix.")
        self.b_salvar.grid(row=0, column=1, rowspan=2, padx=(0, 8))
        self.b_rel = Botao(dir_, "RELATÓRIO", 86, 19, comando=self.abrir_relatorio, tam_fonte=7)
        self.b_rel.grid(row=0, column=2, pady=(0, 1))
        self.b_pasta = Botao(dir_, "PASTA", 86, 19, comando=lambda: abrir_na_pasta(self.ultima_saida), tam_fonte=7)
        self.b_pasta.grid(row=1, column=2, pady=(1, 0))
        Botao(dir_, "PADRÃO", 60, 40, comando=self.voltar_padrao, tam_fonte=7,
              dica="Volta todos os ajustes ao padrão.").grid(row=0, column=3, rowspan=2, padx=(6, 0))
        # forma de onda
        self.c_onda = tk.Canvas(linha, bg=TRILHO, highlightthickness=1, highlightbackground=BORDA, height=62,
                                cursor="hand2")
        self.c_onda.pack(side="left", fill="x", expand=True, padx=6)
        self.c_onda.bind("<Configure>", lambda e: self.preparar_onda())
        self.c_onda.bind("<ButtonPress-1>", self.onda_clique)
        self.c_onda.bind("<B1-Motion>", self.onda_clique)
        self.c_onda.bind("<ButtonRelease-1>", self.onda_soltou)

    # ============================================================ estado/ajustes
    def texto_ref(self) -> str:
        if not self.v_ref:
            return "REF: escolher música…"
        n = os.path.basename(self.v_ref)
        return "REF: " + (n if len(n) <= 26 else n[:25] + "…")

    def mudou(self, *_):
        if self.montando:
            return
        self.atualizar()

    def alvo_mexido(self):
        d = self.seg_destino.get()
        if d in ("streaming", "show") and abs(self.k_alvo.get() - motor.DESTINOS[d][1]) > 0.01:
            self.seg_destino.set("personalizado")
        self.mudou()

    def mudou_destino(self, inicial=False):
        d = self.seg_destino.get()
        lufs = motor.DESTINOS[d][1]
        if lufs is not None and not inicial:
            self.k_alvo.set(lufs)
        if d == "referencia" and not self.v_ref and not inicial:
            self.escolher_ref()
        self.mudou()

    def atualizar(self):
        """Habilita/desabilita controles e marca se os ajustes mudaram desde o processamento."""
        ref = self.seg_destino.get() == "referencia"
        self.k_alvo.ativo(not ref)
        self.b_ref.ativo(ref)
        self.sel_estilo.ativo(not ref)
        eq = self.p_eq.ligado()
        self.k_grave.ativo(eq and not self.b_grave_auto.get())
        self.k_agudo.ativo(eq and not self.b_agudo_auto.get())
        self.b_grave_auto.ativo(eq)
        self.b_agudo_auto.ativo(eq)
        self.k_match.ativo(eq and ref)
        for k in self.k_ms.values():
            k.ativo(eq)
        din = self.p_din.ligado()
        for w in (self.seg_din, self.seg_multi, self.seg_cola):
            w.ativo(din)
        for b in self.b_faixas.values():
            b.ativo(din and self.seg_din.get() != "desligado")
        est = self.p_est.ligado()
        self.b_mono.ativo(est)
        self.k_mono_hz.ativo(est and self.b_mono.get())
        self.k_largura.ativo(est)
        st = self.p_stems.ligado()
        for k in self.k_stems.values():
            k.ativo(st)
        multi = len(self.arquivos) > 1
        if multi:
            self.q_aprox.pack_forget()
            self.q_lista.pack(fill="both", expand=True)
            self.l_qtd.configure(text=str(len(self.arquivos)))
            mex = sum(1 for c in self.arquivos if abs(self.stem_ganhos.get(c, 0)) >= 0.05)
            self.l_resumo_stems.configure(text=f"{mex} com volume mudado\n{len(self.stem_mudos)} no mudo"
                                          + ("" if st else "\n(módulo desligado:\nsoma sem ajustes)"))
        else:
            self.q_lista.pack_forget()
            self.q_aprox.pack(fill="both", expand=True, pady=(4, 0))
        modo = self.seg_lim.get()
        self.l_lim_info.configure(text={"transparente": "o mais limpo", "pegada": "release rápido:\nmais alto e agressivo",
                                        "softclip": "arredonda os picos\nantes do limitador"}[modo])
        # botões do transporte
        pronto = self.resultado is not None and not self.trabalhando
        self.b_play.ativo(pronto and self.player.disponivel)
        self.seg_ab.ativo(pronto)
        self.b_salvar.ativo(pronto)
        self.b_rel.ativo(self.resultado is not None)
        self.b_pasta.ativo(bool(self.ultima_saida))
        self.b_processar.ativo(not self.trabalhando and bool(self.arquivos))
        if pronto and self.montar_cfg(avisar=False) != self.cfg_processada:
            self.b_processar.definir_texto("PROCESSAR ●")
            self.status("Ajustes mudaram: PROCESSAR de novo para ouvir.", AMARELO)
        elif pronto:
            self.b_processar.definir_texto("PROCESSAR")
        if not self.player.disponivel:
            self.l_teclas.configure(text="sem áudio: rode o instalar de novo")

    def status(self, texto, cor=TEXTO_FRACO):
        self.l_status.configure(text=texto, fg=cor)

    def desenhar_progresso(self):
        c = self.c_prog
        c.delete("all")
        w = int(c.cget("width"))
        if self.progresso > 0:
            c.create_rectangle(0, 0, w * self.progresso / 100, 6, fill=DESTAQUE, outline="")

    # ================================================================== arquivos
    def escolher_arquivos(self):
        cs = filedialog.askopenfilenames(title="Escolha a mix (1 arquivo) ou as stems (vários)",
                                         initialdir=self.cfg.get("pasta"), filetypes=TIPOS_AUDIO)
        if cs:
            self.definir_arquivos(list(cs))

    def definir_arquivos(self, arquivos: list[str]):
        self.arquivos = sorted(arquivos, key=lambda c: os.path.basename(c).lower())
        self.cfg["pasta"] = os.path.dirname(self.arquivos[0])
        self.stem_ganhos = {c: 0.0 for c in self.arquivos}
        self.stem_mudos = set()
        if len(self.arquivos) == 1:
            txt = self.arquivos[0]
        else:
            nomes = ", ".join(os.path.basename(c) for c in self.arquivos[:3])
            txt = f"{len(self.arquivos)} STEMS  —  {nomes}" + ("…" if len(self.arquivos) > 3 else "")
        self.l_arquivo.configure(text=txt, fg=TEXTO)
        if self.mixer:
            self.mixer.destroy()
            self.mixer = None
        self.status("Pronto para processar." if self.resultado is None else "Arquivo novo: clique em PROCESSAR.")
        self.mudou()

    def escolher_ref(self):
        c = filedialog.askopenfilename(title="Escolha a música de referência", initialdir=self.cfg.get("pasta"),
                                       filetypes=TIPOS_REF)
        if c:
            self.v_ref = c
            self.b_ref.definir_texto(self.texto_ref())
        self.mudou()

    # ============================================================ mixer de stems
    def abrir_mixer(self):
        if len(self.arquivos) < 2:
            return
        if self.mixer and self.mixer.winfo_exists():
            self.mixer.lift()
            return
        w = tk.Toplevel(self.raiz, bg=FUNDO)
        w.title("Mixer de stems")
        self.mixer = w
        larg = min(1100, max(700, 40 + 92 * len(self.arquivos)))
        w.geometry(f"{larg}x380")
        topo = quadro(w)
        topo.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(topo, text="MIXER DE STEMS", bg=FUNDO, fg=TEXTO, font=fonte(11, "bold")).pack(side="left")
        tk.Label(topo, text="faders ±6 dB · duplo clique = 0 dB · M = mudo", bg=FUNDO, fg=TEXTO_FRACO,
                 font=fonte(8)).pack(side="left", padx=12)
        Botao(topo, "FECHAR", 70, 24, comando=w.destroy, tam_fonte=7.5).pack(side="right")
        Botao(topo, "ZERAR", 70, 24, comando=self.zerar_stems, tam_fonte=7.5).pack(side="right", padx=6)
        cv = tk.Canvas(w, bg=FUNDO, highlightthickness=0, height=300)
        barra = tk.Scrollbar(w, orient="horizontal", command=cv.xview)
        cv.configure(xscrollcommand=barra.set)
        cv.pack(fill="both", expand=True, padx=10)
        if 92 * len(self.arquivos) > larg - 20:
            barra.pack(fill="x", padx=10, pady=(0, 6))
        dentro = tk.Frame(cv, bg=FUNDO)
        cv.create_window((0, 0), window=dentro, anchor="nw")
        dentro.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Shift-MouseWheel>", lambda e: cv.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.faders = {}
        for c in self.arquivos:
            tira = tk.Frame(dentro, bg=PAINEL, highlightbackground=BORDA, highlightthickness=1, width=86, height=292)
            tira.pack(side="left", padx=3, pady=2)
            tira.pack_propagate(False)
            nome = os.path.splitext(os.path.basename(c))[0]
            tk.Label(tira, text=nome, bg=PAINEL, fg=TEXTO, font=fonte(7.5, "bold"), wraplength=78,
                     height=3, justify="center").pack(pady=(4, 0))
            val = tk.Label(tira, text="", bg=TRILHO, fg=DESTAQUE, font=fonte(8, num=True), width=7)
            f = Fader(tira, self.stem_ganhos.get(c, 0.0), altura=170)
            f.mudo = c in self.stem_mudos

            def mudou_fader(v, c=c, val=val):
                self.stem_ganhos[c] = v
                val.configure(text=f"{v:+.1f} dB")
                self.mudou()

            f.ao_mudar = mudou_fader
            f.pack()
            val.configure(text=f"{f.get():+.1f} dB")
            val.pack(pady=(2, 4))
            bm = Botao(tira, "M", 40, 22, modo="liga", led=False, ligado=c in self.stem_mudos, cor=VERMELHO,
                       tam_fonte=8, dica="Mudo")

            def mudou_mudo(c=c, bm=bm, f=f):
                (self.stem_mudos.add if bm.get() else self.stem_mudos.discard)(c)
                f.mudo = bm.get()
                f.desenhar()
                self.mudou()

            bm.comando = mudou_mudo
            bm.pack()
            self.faders[c] = (f, val, bm)

    def zerar_stems(self):
        for c in self.arquivos:
            self.stem_ganhos[c] = 0.0
        self.stem_mudos = set()
        if self.mixer and self.mixer.winfo_exists():
            for c, (f, val, bm) in self.faders.items():
                f.mudo = False
                f.set(0.0)
                val.configure(text="+0.0 dB")
                bm.set(False)
        self.mudou()

    # ============================================================ processamento
    def montar_cfg(self, avisar=True) -> dict | None:
        def aviso(t):
            if avisar:
                messagebox.showwarning("Masterizador", t)
            return None

        if not self.arquivos:
            return aviso("Abra a mix (ou as stems) primeiro: botão ABRIR MIX / STEMS.")
        if avisar:
            faltando = [os.path.basename(c) for c in self.arquivos if not os.path.isfile(c)]
            if faltando:
                return aviso("Não achei estes arquivos (foram movidos?):\n" + "\n".join(faltando))
        destino = self.seg_destino.get()
        ref = self.v_ref or None
        if destino == "referencia" and (not ref or not os.path.isfile(ref)):
            return aviso("Escolha a música de referência (botão REF no módulo DESTINO).")
        eq, din, est, st = self.p_eq.ligado(), self.p_din.ligado(), self.p_est.ligado(), self.p_stems.ligado()
        multi = len(self.arquivos) > 1
        cfg = {
            "destino": destino, "alvo_lufs": self.k_alvo.get(), "teto_tp": self.k_teto.get(),
            "estilo": self.sel_estilo.get(), "referencia": ref if destino == "referencia" else None,
            "grave": (None if self.b_grave_auto.get() else self.k_grave.get()) if eq else 0.0,
            "agudo": (None if self.b_agudo_auto.get() else self.k_agudo.get()) if eq else 0.0,
            "match_eq": self.k_match.get() if eq else 0,
            "ms_mid": {"grave": self.k_ms["mid_grave"].get() if eq else 0.0,
                       "presenca": self.k_ms["mid_presenca"].get() if eq else 0.0,
                       "agudo": self.k_ms["mid_agudo"].get() if eq else 0.0},
            "ms_side": {"grave": self.k_ms["side_grave"].get() if eq else 0.0,
                        "agudo": self.k_ms["side_agudo"].get() if eq else 0.0},
            "dinamico": self.seg_din.get() if din else "desligado",
            "dinamico_faixas": [k for k, b in self.b_faixas.items() if b.get()],
            "multibanda": self.seg_multi.get() if din else "nenhuma",
            "compressao": self.seg_cola.get() if din else "nenhuma",
            "mono_grave": self.b_mono.get() if est else False,
            "mono_grave_hz": self.k_mono_hz.get(), "largura": self.k_largura.get() if est else 100,
            "stems": {k: (w.get() if (st and not multi) else 0.0) for k, w in self.k_stems.items()},
            "saturacao": self.b_sat.get(), "limitador": self.seg_lim.get(),
            "stems_ganhos": dict(self.stem_ganhos) if (multi and st) else {},
            "stems_mudos": sorted(self.stem_mudos) if (multi and st) else [],
        }
        mix = self.arquivos[0] if not multi else list(self.arquivos)
        if avisar:  # guarda as escolhas (o que está nos controles, mesmo com módulo desligado)
            self.cfg.update({
                "destino": destino, "alvo_lufs": self.k_alvo.get(), "teto_tp": self.k_teto.get(),
                "estilo": self.sel_estilo.get(), "referencia": ref,
                "grave": None if self.b_grave_auto.get() else self.k_grave.get(),
                "agudo": None if self.b_agudo_auto.get() else self.k_agudo.get(),
                "grave_valor": self.k_grave.get(), "agudo_valor": self.k_agudo.get(), "match_eq": self.k_match.get(),
                "ms_mid": {"grave": self.k_ms["mid_grave"].get(), "presenca": self.k_ms["mid_presenca"].get(),
                           "agudo": self.k_ms["mid_agudo"].get()},
                "ms_side": {"grave": self.k_ms["side_grave"].get(), "agudo": self.k_ms["side_agudo"].get()},
                "dinamico": self.seg_din.get(), "dinamico_faixas": cfg["dinamico_faixas"],
                "multibanda": self.seg_multi.get(), "compressao": self.seg_cola.get(),
                "mono_grave": self.b_mono.get(), "mono_grave_hz": self.k_mono_hz.get(),
                "largura": self.k_largura.get(), "stems": {k: w.get() for k, w in self.k_stems.items()},
                "saturacao": self.b_sat.get(), "limitador": self.seg_lim.get(),
                "on_eq": eq, "on_din": din, "on_est": est, "on_stems": st})
            gravar_config(self.cfg)
        return {"mix": mix, **cfg}

    def iniciar(self):
        if self.trabalhando:
            return
        cfg = self.montar_cfg()
        if not cfg:
            return
        self.player.parar()
        self.trabalhando = True
        self.resultado = None
        self.esp_vivo = None
        self.progresso = 0
        snapshot = dict(cfg)
        mix = cfg.pop("mix")
        self.desenhar_espectro()
        self.desenhar_medidores()
        self.atualizar()

        def trabalho():
            try:
                r = motor.processar(mix, cfg, lambda pct, txt: self.fila.put(("prog", pct, txt)))
                self.fila.put(("ok", r, snapshot))
            except Exception as e:
                self.fila.put(("erro", e, traceback.format_exc()))

        threading.Thread(target=trabalho, daemon=True).start()

    def salvar(self):
        if not self.resultado:
            return
        if self.montar_cfg(avisar=False) != self.cfg_processada:
            if not messagebox.askyesno("Masterizador", "Você mudou ajustes depois de processar.\n\n"
                                       "Salvar o master como foi processado (sem as mudanças)?\n"
                                       "(Para incluir as mudanças: Não → PROCESSAR.)"):
                return
        try:
            r = motor.salvar(self.resultado)
        except Exception as e:
            messagebox.showerror("Masterizador", f"Não consegui salvar:\n\n{e}")
            return
        self.ultima_saida = r["saida"]
        self.status(f"Salvo: {os.path.basename(r['saida'])} (na pasta da mix).", VERDE)
        if self.janela_rel and self.janela_rel.winfo_exists():
            self.preencher_relatorio()
        self.atualizar()

    def abrir_relatorio(self):
        if not self.resultado:
            return
        if self.janela_rel and self.janela_rel.winfo_exists():
            self.janela_rel.lift()
            self.preencher_relatorio()
            return
        w = tk.Toplevel(self.raiz, bg=FUNDO)
        w.title("Relatório")
        w.geometry("760x560")
        self.janela_rel = w
        t = tk.Text(w, bg=TRILHO, fg=TEXTO, insertbackground=TEXTO, font=fonte(9, num=True), wrap="word",
                    relief="flat", padx=12, pady=10, highlightthickness=0)
        s = tk.Scrollbar(w, command=t.yview)
        t.configure(yscrollcommand=s.set)
        s.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        self.texto_rel = t
        self.preencher_relatorio()

    def preencher_relatorio(self):
        t = self.texto_rel
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", self.resultado["relatorio"])
        t.configure(state="disabled")

    def voltar_padrao(self):
        p = motor.completar({})
        self.montando = True
        self.seg_destino.set(p["destino"])
        self.k_alvo.set(p["alvo_lufs"])
        self.k_teto.set(p["teto_tp"])
        self.sel_estilo.set(p["estilo"])
        self.b_grave_auto.set(True)
        self.b_agudo_auto.set(True)
        for k in (self.k_grave, self.k_agudo, *self.k_ms.values(), *self.k_stems.values()):
            k.set(0.0)
        self.k_match.set(p["match_eq"])
        self.seg_din.set(p["dinamico"])
        for k, b in self.b_faixas.items():
            b.set(k in p["dinamico_faixas"])
        self.seg_multi.set(p["multibanda"])
        self.seg_cola.set(p["compressao"])
        self.b_mono.set(p["mono_grave"])
        self.k_mono_hz.set(p["mono_grave_hz"])
        self.k_largura.set(p["largura"])
        self.seg_lim.set(p["limitador"])
        self.b_sat.set(p["saturacao"])
        for painel in (self.p_eq, self.p_din, self.p_est, self.p_stems):
            painel.liga.set(True)
        self.montando = False
        self.zerar_stems()
        self.mudou()

    # ================================================================ ouvir
    def tocar_parar(self):
        if not self.player.carregado or self.trabalhando:
            return
        if self.player.tocando:
            self.player.parar()
        else:
            try:
                self.player.escolher(self.seg_ab.get())
                self.player.tocar()
            except Exception as e:
                messagebox.showerror("Masterizador", f"Não consegui tocar o áudio:\n\n{e}")

    def lado(self, k):
        if not self.player.carregado:
            return
        self.seg_ab.set(k)
        self.player.escolher(k)
        self.preparar_onda()

    def pular(self, s):
        if self.player.carregado:
            self.player.ir_para(self.player.posicao + s)

    def onda_clique(self, e):
        if not self.player.carregado:
            return
        self.arrastando_onda = True
        w = max(1, self.c_onda.winfo_width())
        self.alvo_onda = max(0.0, min(1.0, e.x / w)) * self.player.duracao
        self.desenhar_onda(self.alvo_onda)

    def onda_soltou(self, _e):
        if self.arrastando_onda and self.player.carregado:
            self.player.ir_para(self.alvo_onda)
        self.arrastando_onda = False

    # ================================================================ desenho
    def preparar_onda(self):
        """Calcula o envelope (mín/máx por coluna) do lado escolhido."""
        if not self.player.carregado:
            self.onda = None
            self.desenhar_onda(0)
            return
        buf = self.player.a if self.seg_ab.get() == "a" else self.player.b
        w = max(50, self.c_onda.winfo_width())
        m = buf.mean(axis=1)
        n = len(m) // w
        if n < 1:
            self.onda = None
            return
        blocos = m[: n * w].reshape(w, n)
        self.onda = (blocos.min(axis=1), blocos.max(axis=1))
        self.c_onda.delete("all")
        self.desenhar_onda(self.player.posicao)

    def desenhar_onda(self, pos: float):
        """A forma de onda é desenhada uma vez (um polígono); a cada quadro só a
        agulha e o tempo se movem."""
        c = self.c_onda
        w, h = c.winfo_width(), c.winfo_height()
        if self.onda is None or not self.player.carregado:
            if not getattr(self, "_onda_vazia", False):
                c.delete("all")
                c.create_text(w / 2, h / 2, text="a forma de onda aparece aqui depois de PROCESSAR",
                              fill=TEXTO_APAGADO, font=fonte(8))
                self._onda_vazia = True
            return
        if getattr(self, "_onda_vazia", True) or not c.find_withtag("agulha"):
            self._onda_vazia = False
            c.delete("all")
            mn, mx = self.onda
            meio = h / 2
            k = h / 2 - 3
            n = min(w, len(mn))
            topo = [v for x in range(n) for v in (x, meio - mx[x] * k)]
            base = [v for x in range(n - 1, -1, -1) for v in (x, meio - mn[x] * k + 1)]
            cor = DESTAQUE if self.seg_ab.get() == "b" else AZUL
            c.create_polygon(topo + base, fill=ui.misturar(cor, "#000000", 0.35), outline=cor)
            c.create_rectangle(0, h - 3, 0, h, fill=cor, outline="", tags="feito")
            c.create_line(0, 0, 0, h, fill=TEXTO, width=2, tags="agulha")
            c.create_text(6, 9, text="", anchor="w", fill=TEXTO, font=fonte(8, num=True), tags="tempo")
            c.create_text(w - 6, 9, text=mmss(self.player.duracao), anchor="e", fill=TEXTO_FRACO,
                          font=fonte(8, num=True))
            lado = "B · MASTER" if self.seg_ab.get() == "b" else "A · ORIGINAL"
            c.create_text(w - 6, h - 10, text=lado, anchor="e", fill=cor, font=fonte(7, "bold"))
        x = int(pos / max(self.player.duracao, 1e-9) * w)
        c.coords("agulha", x, 0, x, h)
        c.coords("feito", 0, h - 3, x, h)
        c.itemconfigure("tempo", text=mmss(pos))

    def x_freq(self, f, w):
        return 40 + (math.log10(f) - math.log10(20)) / (math.log10(20000) - math.log10(20)) * (w - 50)

    def desenhar_espectro(self):
        """Fundo do analisador (grade e curvas médias). O espectro ao vivo é um
        polígono só, atualizado em mover_espectro_vivo()."""
        c = self.c_esp
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50:
            return
        topo, base = 10, h - 18
        self._esp_area = (w, topo, base)
        for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
            x = self.x_freq(f, w)
            c.create_line(x, topo, x, base, fill="#1c1f23")
            c.create_text(x, base + 9, text=(f"{f // 1000}k" if f >= 1000 else str(f)), fill=TEXTO_APAGADO,
                          font=fonte(7, num=True))
        for i in range(1, 6):
            y = topo + (base - topo) * i / 6
            c.create_line(40, y, w - 10, y, fill="#1c1f23")
        c.create_polygon(0, 0, 0, 0, 0, 0, fill="#3a2a12", outline="#8a6020", tags="vivo", state="hidden")
        g = (self.resultado or {}).get("graficos")
        if g:
            for chave, cor, tracejado in (("esp_ref", TEXTO, True), ("esp_antes", AZUL, False),
                                          ("esp_depois", DESTAQUE, False)):
                if not g.get(chave):
                    continue
                fx, db = g[chave]
                meio = (fx >= 200) & (fx <= 2000)
                rel = db - np.mean(db[meio])
                pts = []
                for f_, d in zip(fx, rel):
                    pts += [self.x_freq(f_, w), base - (max(-45, min(15, d)) + 45) / 60 * (base - topo)]
                c.create_line(pts, fill=cor, width=2, dash=(4, 3) if tracejado else None, tags="curva")
        else:
            c.create_text(w / 2, h / 2, text="as curvas de antes/depois aparecem aqui depois de PROCESSAR",
                          fill=TEXTO_APAGADO, font=fonte(8.5), tags="aviso_esp")
        self.mover_espectro_vivo()

    def mover_espectro_vivo(self):
        c = self.c_esp
        if not hasattr(self, "_esp_area") or not c.find_withtag("vivo"):
            return
        if self.esp_vivo is None:
            c.itemconfigure("vivo", state="hidden")
            return
        w, topo, base = self._esp_area
        fx, db = self.esp_vivo
        ys = base - (np.clip(db, -90, 0) + 90) / 90 * (base - topo)
        xs = [self.x_freq(f_, w) for f_ in fx]
        pts = [xs[0], base] + [v for xy in zip(xs, ys.tolist()) for v in xy] + [xs[-1], base]
        c.coords("vivo", *pts)
        c.itemconfigure("vivo", state="normal")
        c.tag_raise("curva")

    def espectro_ao_vivo(self):
        if not self.player.tocando:
            if self.esp_vivo is not None:
                fx, db = self.esp_vivo
                db = db - 3.0
                self.esp_vivo = None if np.max(db) < -90 else (fx, db)
            return
        buf = self.player.a if self.player.lado == "a" else self.player.b
        n = 4096
        i = min(max(0, self.player.pos - n), max(0, len(buf) - n))
        trecho = buf[i:i + n].mean(axis=1) * self.player.ganho[self.player.lado]
        if len(trecho) < n:
            return
        sr = self.player.sr
        esp = np.abs(np.fft.rfft(trecho * np.hanning(n))) / (n / 4)
        f = np.fft.rfftfreq(n, 1 / sr)
        if not hasattr(self, "_bandas_vivo") or self._bandas_vivo[0] != sr:
            grade = np.geomspace(25, min(20000, sr * 0.48), 96)
            idx = [np.flatnonzero((f >= g_ * 2 ** (-1 / 12)) & (f <= g_ * 2 ** (1 / 12))) for g_ in grade]
            idx = [ix if len(ix) else np.array([int(np.argmin(np.abs(f - g_)))]) for ix, g_ in zip(idx, grade)]
            self._bandas_vivo = (sr, grade, idx)
        _, grade, idx = self._bandas_vivo
        db = np.array([20 * np.log10(np.sqrt(np.mean(esp[ix] ** 2)) + 1e-9) for ix in idx])
        if self.esp_vivo is not None:
            db = np.maximum(db, self.esp_vivo[1] - 2.5)  # cai devagar
        self.esp_vivo = (grade, db)

    def ler_medidores(self):
        """Valores dos medidores no ponto atual (do lado que está tocando)."""
        m = self.medidor
        tocando = self.player.tocando
        g = (self.resultado or {}).get("graficos")
        if tocando and g:
            lado = self.player.lado
            med = g["med_master"] if lado == "b" else g["med_original"]
            ganho = 0.0  # nível real do sinal (o 'igualar volume' só muda o que se ouve)
            i = min(int(self.player.posicao / med["passo"]), len(med["lufs_m"]) - 1)
            pico = med["pico"][i] + ganho
            l_, r_ = float(pico[0]), float(pico[-1])
            lm = float(med["lufs_m"][i]) + ganho
            gr = float(g["gr"][min(i, len(g["gr"]) - 1)]) if lado == "b" else 0.0
        else:
            l_ = r_ = -90.0
            lm, gr = -60.0, 0.0
        m["L"] = max(l_, m["L"] - 1.2)
        m["R"] = max(r_, m["R"] - 1.2)
        m["M"] = lm if tocando else max(-60.0, m["M"] - 2)
        m["GR"] = max(gr, m["GR"] - 0.8)
        m["t_hold"] += 1
        if l_ > m["holdL"] or r_ > m["holdR"] or m["t_hold"] > 30:
            m["holdL"], m["holdR"], m["t_hold"] = max(l_, m["L"]), max(r_, m["R"]), 0

    def desenhar_medidores(self):
        """Fundo dos medidores (escalas, títulos e números de antes/depois)."""
        c = self.c_med
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50:
            return
        topo, base = 22, h - 16
        self._med_area = (w, topo, base)
        for d in (0, -6, -12, -24, -36, -48):
            c.create_text(14, self.y_med(d, -48, 0), text=str(d), fill=TEXTO_APAGADO, font=fonte(6.5, num=True))
        for x, rot in ((26, "L"), (44, "R"), (78, "M"), (110, "GR")):
            c.create_rectangle(x, topo, x + 14, base, fill=TRILHO, outline="")
            c.create_text(x + 7, base + 8, text=rot, fill=TEXTO_FRACO, font=fonte(7, "bold"))
        c.create_text(42, 10, text="PICO", fill=TEXTO_FRACO, font=fonte(7, "bold"))
        c.create_text(85, 10, text="LUFS", fill=TEXTO_FRACO, font=fonte(7, "bold"))
        c.create_text(117, 10, text="GR", fill=TEXTO_FRACO, font=fonte(7, "bold"))
        x0 = 146
        r = self.resultado
        lin = topo
        c.create_text(x0, lin, text="LUFS MOMENTÂNEO", anchor="w", fill=TEXTO_FRACO, font=fonte(7, "bold"))
        c.create_text(w - 10, lin + 18, text="—", anchor="e", fill=TEXTO_APAGADO, font=fonte(20, "bold", num=True),
                      tags="lufs_m")
        lin += 50
        if r:
            a, d = r["antes"], r["depois"]

            def fmt(v, casas=1):
                return "—" if v is None else f"{v:.{casas}f}"

            din_a = None if a["curto_p90"] is None else a["curto_p90"] - a["curto_p10"]
            din_d = None if d["curto_p90"] is None else d["curto_p90"] - d["curto_p10"]
            c.create_text(x0 + 92, lin, text="ANTES", anchor="e", fill=AZUL, font=fonte(7, "bold"))
            c.create_text(w - 10, lin, text="DEPOIS", anchor="e", fill=DESTAQUE, font=fonte(7, "bold"))
            for nome, va, vd in (("INTEGRADO", fmt(a["lufs"]), fmt(d["lufs"])),
                                 ("TRUE PEAK", fmt(a["true_peak_db"]), fmt(d["true_peak_db"])),
                                 ("DINÂMICA", fmt(din_a), fmt(din_d)),
                                 ("CORRELAÇÃO", fmt(a["correlacao"], 2), fmt(d["correlacao"], 2))):
                lin += 20
                c.create_text(x0, lin, text=nome, anchor="w", fill=TEXTO_FRACO, font=fonte(7, "bold"))
                c.create_text(x0 + 92, lin, text=va, anchor="e", fill=TEXTO, font=fonte(9, num=True))
                c.create_text(w - 10, lin, text=vd, anchor="e", fill=TEXTO, font=fonte(9, "bold", num=True))
            if r["avisos"]:
                lin += 24
                c.create_text(x0, lin, text=f"⚠ {len(r['avisos'])} aviso(s): veja o RELATÓRIO", anchor="w",
                              fill=AMARELO, font=fonte(7.5, "bold"))
        else:
            c.create_text(x0, lin + 20, text="Os números de antes e\ndepois aparecem aqui\ndepois de PROCESSAR.",
                          anchor="nw", fill=TEXTO_APAGADO, font=fonte(8))
        self.mover_medidores()

    def y_med(self, d, lo, hi):
        _w, topo, base = getattr(self, "_med_area", (0, 22, 200))
        return base - (max(lo, min(hi, d)) - lo) / (hi - lo) * (base - topo)

    def mover_medidores(self):
        """Só as barras e o número mudam a cada quadro (poucos itens)."""
        c = self.c_med
        if not hasattr(self, "_med_area"):
            return
        c.delete("din")
        m = self.medidor
        _w, topo, base = self._med_area

        def barra(x, valor, lo=-48.0, hi=0.0, hold=None):
            y = self.y_med(valor, lo, hi)
            yv, ya = self.y_med(-12, lo, hi), self.y_med(-3, lo, hi)
            if y < base - 0.5:
                c.create_rectangle(x, max(y, yv), x + 14, base, fill=VERDE, outline="", tags="din")
            if y < yv:
                c.create_rectangle(x, max(y, ya), x + 14, yv, fill=AMARELO, outline="", tags="din")
            if y < ya:
                c.create_rectangle(x, y, x + 14, ya, fill=VERMELHO, outline="", tags="din")
            if hold is not None and hold > lo:
                yh = self.y_med(hold, lo, hi)
                c.create_line(x, yh, x + 14, yh, fill=TEXTO, width=2, tags="din")

        barra(26, m["L"], hold=m["holdL"])
        barra(44, m["R"], hold=m["holdR"])
        barra(78, m["M"] + 14, lo=-34, hi=6)  # LUFS momentâneo (0 da escala = -14 LUFS)
        if m["GR"] > 0.05:
            y = topo + min(1.0, m["GR"] / 12) * (base - topo)
            c.create_rectangle(110, topo, 124, y, fill=VERMELHO, outline="", tags="din")
        tocando = self.player.tocando
        c.itemconfigure("lufs_m", text=f"{m['M']:.1f}" if tocando else "—", fill=TEXTO if tocando else TEXTO_APAGADO)

    # ================================================================ ciclo
    def ciclo(self):
        t0 = time.perf_counter()
        try:
            while True:
                msg = self.fila.get_nowait()
                if msg[0] == "prog":
                    self.progresso = msg[1]
                    self.status(msg[2])
                    self.desenhar_progresso()
                elif msg[0] == "ok":
                    r, self.cfg_processada = msg[1], msg[2]
                    self.trabalhando = False
                    self.resultado = r
                    self.player.carregar(r["original"], r["audio"], r["taxa"], r["antes"]["lufs"], r["depois"]["lufs"])
                    self.player.definir_igualar(self.b_igualar.get())
                    self.player.escolher(self.seg_ab.get())
                    self.progresso = 100
                    self.desenhar_progresso()
                    self.status("Pronto! Compare A × B e clique em SALVAR se gostar.", VERDE)
                    self.preparar_onda()
                    self.desenhar_espectro()
                    self.desenhar_medidores()
                    if self.janela_rel and self.janela_rel.winfo_exists():
                        self.preencher_relatorio()
                    self.atualizar()
                elif msg[0] == "erro":
                    self.trabalhando = False
                    self.progresso = 0
                    self.desenhar_progresso()
                    self.status("Deu erro (detalhes na mensagem).", VERMELHO)
                    self.atualizar()
                    messagebox.showerror("Masterizador", f"Não consegui processar:\n\n{msg[1]}\n\n"
                                         "Se não entender, tire um print e mande.")
        except queue.Empty:
            pass
        if self.player.carregado:
            txt = "■  STOP" if self.player.tocando else "▶  PLAY"
            if self.b_play.texto != txt:
                self.b_play.definir_texto(txt)
            if not self.arrastando_onda:
                self.desenhar_onda(self.player.posicao)
        self.espectro_ao_vivo()
        if self.player.tocando or self.esp_vivo is not None or self._vivo_visivel:
            self.mover_espectro_vivo()
            self._vivo_visivel = self.esp_vivo is not None
        self.ler_medidores()
        self.mover_medidores()
        gasto = int((time.perf_counter() - t0) * 1000)
        self.raiz.after(max(15, 50 - gasto), self.ciclo)

    def fechar(self):
        self.player.parar()
        self.raiz.destroy()


# compatibilidade com scripts antigos
Janela = App


def autoteste() -> int:
    """Confere, sem abrir janela, se tudo foi embalado direito (usado na montagem do app).
    O resultado vai para a saída de texto (se houver) e para ~/masterizador_autoteste.txt."""
    import tempfile
    linhas = []
    arq_log = os.path.join(os.path.expanduser("~"), "masterizador_autoteste.txt")

    def print(*a):  # noqa: A001 — no app "sem console" do Windows não existe saída de texto
        linhas.append(" ".join(str(x) for x in a))
        if sys.stdout is not None:
            sys.stdout.write(linhas[-1] + "\n")
        with open(arq_log, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")

    import soundfile as sf
    import player
    sr = 44100
    t = np.arange(sr * 8) / sr
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.default_rng(0).normal(size=len(t))
    with tempfile.TemporaryDirectory() as d:
        arq = os.path.join(d, "teste.wav")
        sf.write(arq, np.stack([x, x * 0.9], axis=1), sr, subtype="PCM_24")
        r = motor.processar(arq, {"destino": "streaming", "alvo_lufs": -14, "multibanda": "leve",
                                  "dinamico": "leve", "stems": {"voz": 1.0}})
        mp3_ok = "MP3" in sf.available_formats()
    lufs = r["depois"]["lufs"]
    print(f"master: {lufs:.2f} LUFS, true peak {r['depois']['true_peak_db']:.2f} dBTP")
    print(f"sounddevice: {'ok' if player.sd is not None else 'FALTANDO ' + str(player.ERRO)}")
    print(f"leitura de MP3: {'ok' if mp3_ok else 'não'}")
    print(f"tkinter: Tk {tk.TkVersion}")
    ok = abs(lufs + 14) < 0.5 and player.sd is not None
    print("AUTOTESTE OK" if ok else "AUTOTESTE FALHOU")
    return 0 if ok else 1


def main():
    if "--autoteste" in sys.argv:
        sys.exit(autoteste())
    raiz = tk.Tk()
    app = App(raiz)
    # arquivos passados na linha de comando: já carrega e processa
    arquivos = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if arquivos:
        app.definir_arquivos(arquivos)
        raiz.after(400, app.iniciar)
    raiz.mainloop()


if __name__ == "__main__":
    main()
