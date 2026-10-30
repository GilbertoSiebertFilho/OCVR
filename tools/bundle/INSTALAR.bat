@echo off
setlocal EnableExtensions
title Olds College apps - instalacao
cd /d "%~dp0"
echo.
echo   ==============================================================
echo     Olds College apps: OCVR, OCCD e NDVI Map Viewer
echo     Instalacao (uma vez so). Pasta: %CD%
echo   ==============================================================
echo.

rem ---- a pasta: OneDrive sincroniza os milhares de arquivos do ambiente Python, o que deixa tudo lento
echo %CD% | find /i "OneDrive" >nul && (
  echo   AVISO: esta pasta fica dentro do OneDrive. Recomendado: mover a pasta OC_Apps para C:\OC_Apps
  echo   e rodar INSTALAR.bat de la. Feche esta janela para mover, ou aperte uma tecla para seguir assim mesmo.
  pause >nul
)

rem ---- 1. Python 3.12 ou mais novo (o atalho da Microsoft Store nao serve: e so um link para a loja)
set "PY="
for %%V in (314 313 312) do (
  if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
  if not defined PY if exist "%ProgramFiles%\Python%%V\python.exe" set "PY=%ProgramFiles%\Python%%V\python.exe"
)
if not defined PY (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)"') do set "PY=%%P"
  )
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "PY=%%P"
  )
)
if defined PY goto :havepython

echo   Python nao encontrado: baixando o Python 3.14.3 oficial de python.org (30 MB) ...
set "PYINST=%TEMP%\python-3.14.3-amd64.exe"
curl.exe -L --fail -o "%PYINST%" "https://www.python.org/ftp/python/3.14.3/python-3.14.3-amd64.exe"
if errorlevel 1 goto :nodownload
echo   Instalando o Python para este usuario (sem precisar de administrador; 1-2 minutos) ...
start "" /wait "%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Shortcuts=0
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PY goto :noinstall

:havepython
for /f "delims=" %%V in ('"%PY%" -c "import sys; print(sys.version.split()[0])"') do set "PYV=%%V"
echo   Python %PYV%: %PY%
echo.

rem ---- 2. o ambiente .venv com as versoes testadas (constraints.txt)
if not exist ".venv\Scripts\python.exe" (
  echo   Criando o ambiente .venv ...
  "%PY%" -m venv .venv
  if errorlevel 1 goto :novenv
)
echo   Instalando os pacotes (cerca de 600 MB, 3-8 minutos na primeira vez) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements_all.txt -c constraints.txt
if errorlevel 1 goto :pipfailed
".venv\Scripts\python.exe" -c "import geopandas, rasterio, statsmodels, sklearn, skimage, fitz, streamlit, reportlab; print('  pacotes OK')"
if errorlevel 1 goto :pipfailed

rem ---- 3. atalhos na area de trabalho
if defined OC_NO_SHORTCUTS goto :done_shortcuts
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcuts.ps1" "%~dp0."
if errorlevel 1 echo   (Os atalhos nao puderam ser criados: use OCVR.bat, OCCD.bat e NDVI.bat desta pasta.)

:done_shortcuts
echo.
echo   ==============================================================
echo     Pronto. Abra pelos atalhos da area de trabalho
echo     (OCVR, OCCD, NDVI Map Viewer) ou pelos .bat desta pasta.
echo   ==============================================================
echo.
pause
exit /b 0

:nodownload
echo.
echo   Nao foi possivel baixar o Python (sem internet, ou bloqueado pela rede).
echo   Instale manualmente: https://www.python.org/downloads/  (marque "Add python.exe to PATH")
echo   e rode INSTALAR.bat de novo.
pause
exit /b 1
:noinstall
echo.
echo   O instalador do Python terminou, mas o Python nao apareceu em %LOCALAPPDATA%\Programs\Python.
echo   A TI pode ter bloqueado a instalacao. Instale pelo Software Center / python.org e rode INSTALAR.bat de novo.
pause
exit /b 1
:novenv
echo.
echo   Nao foi possivel criar o ambiente .venv nesta pasta (permissao?). Mova a pasta para C:\OC_Apps e tente de novo.
pause
exit /b 1
:pipfailed
echo.
echo   A instalacao dos pacotes falhou (veja as mensagens acima). Rode INSTALAR.bat de novo;
echo   se repetir, copie a mensagem de erro.
pause
exit /b 1
