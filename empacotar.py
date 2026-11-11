"""
empacotar.py — gera ../masterizador.zip para mandar para o Mac.

- Os .command vão com fim de linha Unix (LF) e permissão de execução
  marcada no zip (o Utilitário de Arquivo do Mac respeita isso).
- Não inclui os testes, o .bat do Windows, caches nem este arquivo.
Rode: python empacotar.py
"""

import os
import time
import zipfile

PASTA = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(os.path.dirname(PASTA), "masterizador.zip")
ARQUIVOS = ["LEIA-ME.txt", "instalar.command", "masterizar.command", "janela.py", "motor.py", "dsp.py",
            "stems.py", "player.py", "ui_widgets.py", "requirements.txt"]


def main():
    data = time.localtime()[:6]
    with zipfile.ZipFile(DESTINO, "w", zipfile.ZIP_DEFLATED) as z:
        for nome in ARQUIVOS:
            with open(os.path.join(PASTA, nome), "rb") as f:
                dados = f.read()
            executavel = nome.endswith(".command")
            if executavel or nome.endswith((".py", ".txt")):
                dados = dados.replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(f"masterizador/{nome}", date_time=data)
            info.create_system = 3  # Unix: para o Mac ler as permissões
            info.external_attr = ((0o100755 if executavel else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, dados)
    print(f"Pronto: {DESTINO}")


if __name__ == "__main__":
    main()
