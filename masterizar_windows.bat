@echo off
rem Abre a janela do masterizador no Windows (para testar).
cd /d "%~dp0"
python janela.py
if errorlevel 1 pause
