@echo off
title NDVI Map Viewer
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente ainda nao foi instalado: rodando INSTALAR.bat primeiro ...
  call "%~dp0INSTALAR.bat" || exit /b 1
)
cd /d "%~dp0NDVIMAP"
echo NDVI Map Viewer: na primeira vez salve o client id e o secret do Copernicus no painel 1.
"%~dp0.venv\Scripts\python.exe" -m server.app %*
pause
