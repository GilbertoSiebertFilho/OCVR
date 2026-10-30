@echo off
rem OCVR - one-time setup on a new computer: a private Python environment (.venv) with the packages of requirements.txt.
rem Needs Python 3.12 or newer from python.org (tick "Add python.exe to PATH" in the installer). Takes 2-5 minutes.
cd /d "%~dp0"
set PY=
where py >nul 2>nul && (py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set PY=py -3)
if not defined PY (
  where python >nul 2>nul && (python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set PY=python)
)
if not defined PY (
  echo Python 3.12 or newer was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run setup.bat again.
  pause
  exit /b 1
)
echo Using:
%PY% --version
if not exist ".venv\Scripts\python.exe" (
  echo Creating the environment in .venv ...
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
echo Installing the packages (geopandas, rasterio, statsmodels, streamlit, ...) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package installation failed: see the messages above. & pause & exit /b 1)
".venv\Scripts\python.exe" -c "import geopandas, rasterio, statsmodels, streamlit, reportlab; print('OCVR environment ready')"
echo.
echo Done. Start the app with run_app.bat.
pause
