@echo off
rem OCVR - Olds College Variable-Rate economics: start the Streamlit app (it opens in the browser at http://localhost:8501)
rem On a new computer run setup.bat once first; it makes the .venv environment this file uses.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app\app.py
) else (
  python -m streamlit run app\app.py
)
pause
