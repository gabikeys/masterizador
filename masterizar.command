#!/bin/bash
# Abre a janela do masterizador.
cd "$(dirname "$0")"

PY=""
for c in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 /usr/local/bin/python3; do
  if [ -x "$c" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Não achei o Python instalado pelo python.org."
  echo "Instale em https://www.python.org/downloads/macos/ e depois rode instalar.command."
  read -p "Aperte Enter para fechar."
  exit 1
fi

if ! "$PY" janela.py; then
  echo ""
  echo "O programa fechou com erro. Tire um print desta janela e mande."
  read -p "Aperte Enter para fechar."
fi
