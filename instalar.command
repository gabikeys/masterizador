#!/bin/bash
# Instala as bibliotecas de áudio do masterizador (rode uma vez só).
cd "$(dirname "$0")"

PY=""
for c in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 /usr/local/bin/python3; do
  if [ -x "$c" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo ""
  echo "Não achei o Python instalado pelo python.org."
  echo "Instale em https://www.python.org/downloads/macos/ e rode este arquivo de novo."
  echo ""
  read -p "Aperte Enter para fechar."
  exit 1
fi

echo "Usando: $PY ($("$PY" --version 2>&1))"
echo ""
"$PY" -m pip install --upgrade pip
if "$PY" -m pip install -r requirements.txt; then
  echo ""
  echo "Pronto! Agora é só dar dois cliques em masterizar.command."
else
  echo ""
  echo "Deu erro na instalação. Tire um print desta janela e mande."
  echo "Dica: se falar de 'certificate', rode o 'Install Certificates.command'"
  echo "que fica em Aplicativos > Python 3.x e tente de novo."
fi
echo ""
read -p "Aperte Enter para fechar."
