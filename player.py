"""
player.py — toca a mix original (A) e o master (B) para comparar antes de salvar.

A troca A/B é instantânea e continua do mesmo ponto. Com 'igualar volume', os
dois tocam no mesmo loudness (o mais alto é abaixado), para comparar o som e
não o volume. Usa a biblioteca sounddevice (se não estiver instalada, o resto
do programa funciona normalmente, só sem ouvir).
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
    ERRO = None
except Exception as e:  # biblioteca ausente ou sem placa de som
    sd = None
    ERRO = str(e)


def _estereo32(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[:, :2], dtype=np.float32)


class Player:
    def __init__(self):
        self.a = self.b = None
        self.sr = 44100
        self.pos = 0
        self.lado = "b"
        self.ganho = {"a": 1.0, "b": 1.0}
        self.lufs = {"a": None, "b": None}
        self.igualar = True
        self.stream = None
        self.trava = threading.Lock()

    @property
    def disponivel(self) -> bool:
        return sd is not None

    @property
    def carregado(self) -> bool:
        return self.a is not None

    @property
    def duracao(self) -> float:
        return 0.0 if self.a is None else len(self.a) / self.sr

    @property
    def posicao(self) -> float:
        return self.pos / self.sr

    @property
    def tocando(self) -> bool:
        return self.stream is not None and self.stream.active

    def carregar(self, original: np.ndarray, master: np.ndarray, sr: int, lufs_a: float | None, lufs_b: float | None):
        self.parar()
        with self.trava:
            self.a, self.b, self.sr = _estereo32(original), _estereo32(master), sr
            self.pos = 0
            self.lufs = {"a": lufs_a, "b": lufs_b}
            self._calcular_ganhos()

    def definir_igualar(self, sim: bool):
        self.igualar = sim
        self._calcular_ganhos()

    def _calcular_ganhos(self):
        la, lb = self.lufs["a"], self.lufs["b"]
        if self.igualar and la is not None and lb is not None:
            alvo = min(la, lb)  # abaixa o mais alto: nunca estoura
            self.ganho = {"a": 10 ** ((alvo - la) / 20), "b": 10 ** ((alvo - lb) / 20)}
        else:
            self.ganho = {"a": 1.0, "b": 1.0}

    def escolher(self, lado: str):
        self.lado = "a" if lado == "a" else "b"

    def ir_para(self, segundos: float):
        with self.trava:
            if self.a is not None:
                self.pos = int(max(0.0, min(segundos, self.duracao)) * self.sr)

    def _callback(self, saida, quadros, _tempo, _status):
        with self.trava:
            buf = self.a if self.lado == "a" else self.b
            if buf is None:
                saida[:] = 0
                raise sd.CallbackStop
            n = max(0, min(quadros, len(buf) - self.pos))
            saida[:n] = buf[self.pos:self.pos + n] * self.ganho[self.lado]
            saida[n:] = 0
            self.pos += n
            if n < quadros:
                raise sd.CallbackStop

    def tocar(self):
        if sd is None:
            raise RuntimeError(f"a biblioteca de áudio (sounddevice) não está disponível: {ERRO}")
        if self.a is None or self.tocando:
            return
        self._fechar()
        if self.pos >= len(self.a) - 1:
            self.pos = 0
        self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32", callback=self._callback)
        self.stream.start()

    def parar(self):
        self._fechar()

    def _fechar(self):
        s, self.stream = self.stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
