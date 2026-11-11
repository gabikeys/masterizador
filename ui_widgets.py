"""
ui_widgets.py — componentes desenhados à mão (Canvas) no estilo de plugin de estúdio.

Tudo é desenhado no Canvas porque no Mac os botões nativos do Tk ignoram cores.
Componentes: Knob, Botao (ação ou liga/desliga com LED), Segmentado, Seletor, Painel.
"""

from __future__ import annotations

import math
import sys
import tkinter as tk

# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------

FUNDO = "#141619"
PAINEL = "#1e2125"
PAINEL_TOPO = "#272b30"
BORDA = "#343a41"
TEXTO = "#d9dde2"
TEXTO_FRACO = "#7f8892"
TEXTO_APAGADO = "#4a5058"
DESTAQUE = "#f2a33a"          # âmbar
DESTAQUE_ESCURO = "#6b4a1c"
VERDE = "#3fd27a"
VERMELHO = "#ff5252"
AMARELO = "#ffd43b"
AZUL = "#5aa9ff"
CORPO_KNOB = "#2d3136"
TRILHO = "#0f1113"

MAC = sys.platform == "darwin"
FONTE = "Helvetica Neue" if MAC else "Segoe UI"
FONTE_NUM = "Menlo" if MAC else "Consolas"
_ESC = 1.25 if MAC else 1.0  # o Tk do Mac desenha fontes menores


def fonte(tam: float, peso: str = "normal", num: bool = False):
    return (FONTE_NUM if num else FONTE, int(round(tam * _ESC)), peso)


def retangulo_redondo(c: tk.Canvas, x1, y1, x2, y2, r=6, **kw):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def misturar(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def _roda(evento) -> int:
    if getattr(evento, "num", None) == 4:
        return 1
    if getattr(evento, "num", None) == 5:
        return -1
    return 1 if evento.delta > 0 else -1


# ---------------------------------------------------------------------------
# Knob
# ---------------------------------------------------------------------------

class Knob(tk.Canvas):
    """Knob giratório: arraste para cima/baixo (Shift = fino), roda do mouse,
    duplo clique volta ao padrão."""

    def __init__(self, pai, rotulo: str, de: float, ate: float, valor: float, passo: float,
                 padrao: float | None = None, formato: str = "{:+.1f}", unidade: str = "",
                 bipolar: bool | None = None, tam: int = 44, ao_mudar=None, dica: str = "",
                 centro: float = 0.0):
        self.tam = tam
        w, h = tam + 22, tam + 38
        super().__init__(pai, width=w, height=h, bg=pai.cget("bg"), highlightthickness=0, cursor="sb_v_double_arrow")
        self.rotulo, self.de, self.ate, self.passo = rotulo, de, ate, passo
        self.padrao = valor if padrao is None else padrao
        self.formato, self.unidade = formato, unidade
        self.centro = centro
        self.bipolar = (de < centro < ate) if bipolar is None else bipolar
        self.ao_mudar = ao_mudar
        self.habilitado = True
        self.valor = self._ajustar(valor)
        self._y0 = self._v0 = 0.0
        self.bind("<ButtonPress-1>", self._apertou)
        self.bind("<B1-Motion>", self._arrastou)
        self.bind("<Double-Button-1>", lambda e: self._mudar(self.padrao))
        self.bind("<MouseWheel>", lambda e: self._mudar(self.valor + _roda(e) * self.passo))
        self.bind("<Button-4>", lambda e: self._mudar(self.valor + self.passo))
        self.bind("<Button-5>", lambda e: self._mudar(self.valor - self.passo))
        if dica:
            Dica(self, dica)
        self.desenhar()

    def _ajustar(self, v: float) -> float:
        v = max(self.de, min(self.ate, v))
        return round(round(v / self.passo) * self.passo, 6)

    def _apertou(self, e):
        self._y0, self._v0 = e.y_root, self.valor

    def _arrastou(self, e):
        if not self.habilitado:
            return
        faixa = (self.ate - self.de) * (0.15 if e.state & 0x0001 else 1.0)
        self._mudar(self._v0 + (self._y0 - e.y_root) * faixa / 160.0)

    def _mudar(self, v: float):
        if not self.habilitado:
            return
        v = self._ajustar(v)
        if v != self.valor:
            self.valor = v
            self.desenhar()
            if self.ao_mudar:
                self.ao_mudar(v)

    def get(self) -> float:
        return self.valor

    def set(self, v: float):
        self.valor = self._ajustar(v)
        self.desenhar()

    def ativo(self, sim: bool):
        self.habilitado = sim
        self.configure(cursor="sb_v_double_arrow" if sim else "arrow")
        self.desenhar()

    def desenhar(self):
        self.delete("all")
        w = int(self.cget("width"))
        cx, cy, r = w / 2, 16 + self.tam / 2, self.tam / 2
        cor = DESTAQUE if self.habilitado else TEXTO_APAGADO
        self.create_text(cx, 7, text=self.rotulo, fill=TEXTO_FRACO if self.habilitado else TEXTO_APAGADO,
                         font=fonte(7.5, "bold"))
        f = (self.valor - self.de) / (self.ate - self.de)
        ang = 225 - 270 * f
        bb = (cx - r, cy - r, cx + r, cy + r)
        self.create_arc(bb, start=225, extent=-270, style="arc", outline=TRILHO, width=4)
        if self.bipolar:
            f0 = (self.centro - self.de) / (self.ate - self.de)
            a0 = 225 - 270 * f0
            if abs(ang - a0) > 0.5:
                self.create_arc(bb, start=a0, extent=ang - a0, style="arc", outline=cor, width=4)
        elif f > 0.002:
            self.create_arc(bb, start=225, extent=ang - 225, style="arc", outline=cor, width=4)
        ri = r - 6
        self.create_oval(cx - ri, cy - ri + 1, cx + ri, cy + ri + 1, fill="#0b0c0e", outline="")
        self.create_oval(cx - ri, cy - ri, cx + ri, cy + ri, fill=CORPO_KNOB, outline="#3c4248")
        self.create_oval(cx - ri + 3, cy - ri + 3, cx + ri - 3, cy + ri - 3, fill="#33383e", outline="")
        rad = math.radians(ang)
        self.create_line(cx + (ri - 12) * math.cos(rad) * 0.2, cy - (ri - 12) * math.sin(rad) * 0.2,
                         cx + (ri - 3) * math.cos(rad), cy - (ri - 3) * math.sin(rad),
                         fill=TEXTO if self.habilitado else TEXTO_APAGADO, width=2.5, capstyle="round")
        txt = self.formato.format(self.valor) + self.unidade
        self.create_text(cx, cy + r + 10, text=txt, fill=TEXTO if self.habilitado else TEXTO_APAGADO,
                         font=fonte(8, num=True))


class Fader(tk.Canvas):
    """Fader vertical (mixer de stems): arraste, roda do mouse, duplo clique = 0 dB."""

    def __init__(self, pai, valor=0.0, de=-6.0, ate=6.0, passo=0.5, altura=190, ao_mudar=None):
        super().__init__(pai, width=44, height=altura, bg=pai.cget("bg"), highlightthickness=0,
                         cursor="sb_v_double_arrow")
        self.de, self.ate, self.passo, self.ao_mudar = de, ate, passo, ao_mudar
        self.valor = valor
        self.mudo = False
        self.bind("<ButtonPress-1>", lambda e: self._ir(e.y))
        self.bind("<B1-Motion>", lambda e: self._ir(e.y))
        self.bind("<Double-Button-1>", lambda e: self._mudar(0.0))
        self.bind("<MouseWheel>", lambda e: self._mudar(self.valor + _roda(e) * self.passo))
        self.desenhar()

    def _y(self, v):
        h = int(self.cget("height"))
        return 12 + (self.ate - v) / (self.ate - self.de) * (h - 24)

    def _ir(self, y):
        h = int(self.cget("height"))
        self._mudar(self.ate - (y - 12) / (h - 24) * (self.ate - self.de))

    def _mudar(self, v):
        v = round(round(max(self.de, min(self.ate, v)) / self.passo) * self.passo, 6)
        if v != self.valor:
            self.valor = v
            self.desenhar()
            if self.ao_mudar:
                self.ao_mudar(v)

    def set(self, v):
        self.valor = v
        self.desenhar()

    def get(self):
        return self.valor

    def desenhar(self):
        self.delete("all")
        h = int(self.cget("height"))
        for v in (6, 3, 0, -3, -6):
            y = self._y(v)
            self.create_line(4, y, 12, y, fill=TEXTO_FRACO if v == 0 else BORDA)
            self.create_line(32, y, 40, y, fill=TEXTO_FRACO if v == 0 else BORDA)
        self.create_rectangle(20, 10, 24, h - 10, fill=TRILHO, outline="")
        y0, y = self._y(0), self._y(self.valor)
        if abs(y - y0) > 1:
            self.create_rectangle(20, min(y, y0), 24, max(y, y0),
                                  fill=TEXTO_APAGADO if self.mudo else DESTAQUE, outline="")
        retangulo_redondo(self, 9, y - 9, 35, y + 9, r=3, fill="#3b4148", outline="#5a626b")
        self.create_line(11, y, 33, y, fill=TEXTO if not self.mudo else TEXTO_FRACO, width=2)


# ---------------------------------------------------------------------------
# Botões
# ---------------------------------------------------------------------------

class Botao(tk.Canvas):
    """Botão desenhado. modo='acao' (clique) ou 'liga' (liga/desliga com LED)."""

    def __init__(self, pai, texto: str, largura: int = 80, altura: int = 26, modo: str = "acao",
                 ligado: bool = False, comando=None, cor: str = DESTAQUE, led: bool | None = None,
                 tam_fonte: float = 8, destaque: bool = False, dica: str = ""):
        super().__init__(pai, width=largura, height=altura, bg=pai.cget("bg"), highlightthickness=0, cursor="hand2")
        self.texto, self.modo, self.ligado, self.comando = texto, modo, ligado, comando
        self.cor, self.tam_fonte, self.destaque = cor, tam_fonte, destaque
        self.led = (modo == "liga") if led is None else led
        self.habilitado, self.sobre, self.apertado = True, False, False
        self.bind("<Enter>", lambda e: self._estado(sobre=True))
        self.bind("<Leave>", lambda e: self._estado(sobre=False, apertado=False))
        self.bind("<ButtonPress-1>", lambda e: self._estado(apertado=True))
        self.bind("<ButtonRelease-1>", self._soltou)
        if dica:
            Dica(self, dica)
        self.desenhar()

    def _estado(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.desenhar()

    def _soltou(self, e):
        dentro = 0 <= e.x <= int(self.cget("width")) and 0 <= e.y <= int(self.cget("height"))
        self.apertado = False
        if dentro and self.habilitado:
            if self.modo == "liga":
                self.ligado = not self.ligado
            self.desenhar()
            if self.comando:
                self.comando()
        else:
            self.desenhar()

    def set(self, ligado: bool):
        self.ligado = ligado
        self.desenhar()

    def get(self) -> bool:
        return self.ligado

    def ativo(self, sim: bool):
        self.habilitado = sim
        self.configure(cursor="hand2" if sim else "arrow")
        self.desenhar()

    def definir_texto(self, t: str):
        self.texto = t
        self.desenhar()

    def desenhar(self):
        self.delete("all")
        w, h = int(self.cget("width")), int(self.cget("height"))
        aceso = self.ligado and self.modo != "acao"
        if not self.habilitado:
            fundo, borda, cor_txt = "#1a1d20", "#2a2e33", TEXTO_APAGADO
        elif self.destaque:
            base = self.cor
            fundo = misturar(base, "#000000", 0.25 if self.apertado else (0.0 if self.sobre else 0.12))
            borda, cor_txt = misturar(base, "#ffffff", 0.2), "#16120a"
        elif aceso and not self.led:
            fundo = misturar(self.cor, "#000000", 0.55 if self.apertado else 0.45)
            borda, cor_txt = self.cor, "#ffffff"
        else:
            fundo = "#191b1e" if self.apertado else ("#30353b" if self.sobre else "#2a2e33")
            borda, cor_txt = "#41474e", TEXTO if (aceso or self.modo == "acao") else TEXTO_FRACO
        retangulo_redondo(self, 1, 1, w - 1, h - 1, r=5, fill=fundo, outline=borda)
        tx = w / 2
        if self.led:
            lx, ly = 11, h / 2
            cor_led = self.cor if (self.ligado and self.habilitado) else "#3a2f22" if self.cor == DESTAQUE else "#22302a"
            if self.ligado and self.habilitado:
                self.create_oval(lx - 6, ly - 6, lx + 6, ly + 6, fill=misturar(self.cor, fundo, 0.65), outline="")
            self.create_oval(lx - 3.5, ly - 3.5, lx + 3.5, ly + 3.5, fill=cor_led, outline="#000000")
            tx = (w + 16) / 2
        self.create_text(tx, h / 2, text=self.texto, fill=cor_txt, font=fonte(self.tam_fonte, "bold"))


class Segmentado(tk.Frame):
    """Grupo de botões em que só um fica aceso."""

    def __init__(self, pai, opcoes: list[tuple[str, str]], valor: str, comando=None,
                 largura: int = 60, altura: int = 24, vertical: bool = False, colunas: int | None = None,
                 tam_fonte: float = 7.5):
        super().__init__(pai, bg=pai.cget("bg"))
        self.valor, self.comando, self.botoes = valor, comando, {}
        colunas = colunas or (1 if vertical else len(opcoes))
        for i, (chave, texto) in enumerate(opcoes):
            b = Botao(self, texto, largura, altura, modo="liga", led=False, ligado=(chave == valor),
                      comando=lambda k=chave: self._escolheu(k), tam_fonte=tam_fonte)
            b.grid(row=i // colunas, column=i % colunas, padx=1, pady=1)
            self.botoes[chave] = b

    def _escolheu(self, k):
        self.set(k)
        if self.comando:
            self.comando(k)

    def set(self, k):
        self.valor = k
        for chave, b in self.botoes.items():
            b.set(chave == k)

    def get(self) -> str:
        return self.valor

    def ativo(self, sim: bool):
        for b in self.botoes.values():
            b.ativo(sim)


class Seletor(tk.Canvas):
    """◀ texto ▶ para escolher entre várias opções."""

    def __init__(self, pai, opcoes: list[tuple[str, str]], valor: str, largura: int = 180, comando=None):
        super().__init__(pai, width=largura, height=24, bg=pai.cget("bg"), highlightthickness=0, cursor="hand2")
        self.opcoes, self.comando, self.habilitado = opcoes, comando, True
        self.i = next((i for i, (k, _) in enumerate(opcoes) if k == valor), 0)
        self.bind("<ButtonRelease-1>", self._clique)
        self.bind("<MouseWheel>", lambda e: self._passo(-_roda(e)))
        self.desenhar()

    def _clique(self, e):
        self._passo(-1 if e.x < int(self.cget("width")) / 2 else 1)

    def _passo(self, d):
        if not self.habilitado:
            return
        self.i = (self.i + d) % len(self.opcoes)
        self.desenhar()
        if self.comando:
            self.comando(self.get())

    def get(self) -> str:
        return self.opcoes[self.i][0]

    def set(self, k):
        self.i = next((i for i, (c, _) in enumerate(self.opcoes) if c == k), 0)
        self.desenhar()

    def ativo(self, sim: bool):
        self.habilitado = sim
        self.desenhar()

    def desenhar(self):
        self.delete("all")
        w = int(self.cget("width"))
        cor = TEXTO if self.habilitado else TEXTO_APAGADO
        retangulo_redondo(self, 1, 1, w - 1, 23, r=5, fill=TRILHO, outline=BORDA)
        self.create_text(12, 12, text="◀", fill=DESTAQUE if self.habilitado else TEXTO_APAGADO, font=fonte(7))
        self.create_text(w - 12, 12, text="▶", fill=DESTAQUE if self.habilitado else TEXTO_APAGADO, font=fonte(7))
        self.create_text(w / 2, 12, text=self.opcoes[self.i][1], fill=cor, font=fonte(8, "bold"))


class Painel(tk.Frame):
    """Módulo da cadeia: faixa de título (com botão liga/desliga opcional) e corpo."""

    def __init__(self, pai, titulo: str, largura: int, com_liga: bool = False, ligado: bool = True, ao_ligar=None,
                 height: int | None = None):
        super().__init__(pai, bg=BORDA, width=largura, **({"height": height} if height else {}))
        self.grid_propagate(False)
        self.pack_propagate(False)
        topo = tk.Frame(self, bg=PAINEL_TOPO, height=26)
        topo.pack(fill="x", padx=1, pady=(1, 0))
        topo.pack_propagate(False)
        tk.Label(topo, text=titulo, bg=PAINEL_TOPO, fg=TEXTO, font=fonte(8.5, "bold")).pack(side="left", padx=8)
        self.liga = None
        if com_liga:
            self.liga = Botao(topo, "ON", 46, 18, modo="liga", ligado=ligado, comando=ao_ligar, tam_fonte=7,
                              dica="Liga/desliga este módulo (bypass)")
            self.liga.pack(side="right", padx=5)
        self.corpo = tk.Frame(self, bg=PAINEL)
        self.corpo.pack(fill="both", expand=True, padx=1, pady=(0, 1))

    def ligado(self) -> bool:
        return True if self.liga is None else self.liga.get()


def rotulo(pai, texto, tam=7.5, cor=TEXTO_FRACO, peso="bold", **kw):
    return tk.Label(pai, text=texto, bg=pai.cget("bg"), fg=cor, font=fonte(tam, peso), **kw)


class Dica:
    """Balãozinho de ajuda ao parar o mouse em cima."""

    def __init__(self, w, texto):
        self.w, self.texto, self.janela, self.id = w, texto, None, None
        w.bind("<Enter>", self._agendar, add="+")
        w.bind("<Leave>", self._fechar, add="+")
        w.bind("<ButtonPress>", self._fechar, add="+")

    def _agendar(self, _e):
        self.id = self.w.after(600, self._abrir)

    def _abrir(self):
        if self.janela:
            return
        x, y = self.w.winfo_rootx() + 10, self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.janela = tk.Toplevel(self.w)
        self.janela.wm_overrideredirect(True)
        self.janela.wm_geometry(f"+{x}+{y}")
        tk.Label(self.janela, text=self.texto, bg="#30353b", fg=TEXTO, font=fonte(8), justify="left",
                 wraplength=260, padx=8, pady=5, relief="solid", bd=1).pack()

    def _fechar(self, _e=None):
        if self.id:
            self.w.after_cancel(self.id)
            self.id = None
        if self.janela:
            self.janela.destroy()
            self.janela = None
