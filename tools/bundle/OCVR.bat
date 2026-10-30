@echo off
title OCVR - Olds College Variable-Rate economics
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente ainda nao foi instalado: rodando INSTALAR.bat primeiro ...
  call "%~dp0INSTALAR.bat" || exit /b 1
)
cd /d "%~dp0OCVR"
echo OCVR abre no navegador em http://localhost:8501  (feche esta janela para parar o app)
"%~dp0.venv\Scripts\python.exe" -m streamlit run app\app.py --server.port 8501
pause
