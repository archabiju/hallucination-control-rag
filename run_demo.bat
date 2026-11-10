@echo off
REM TrustRAG one-command demo for Windows (CMD or PowerShell).   run_demo.bat
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not exist demo\outputs mkdir demo\outputs

echo == 1/4 Environment and model/config versions ==
python demo\record_versions.py > demo\outputs\versions.txt
type demo\outputs\versions.txt
if errorlevel 1 goto :fail

echo.
echo == 2/4 Build or reuse the FAISS index ==
python src\pipeline.py
if errorlevel 1 goto :fail

echo.
echo == 3/4 Demo queries (live pipeline + tri-state demonstration) ==
python demo\demo_run.py
if errorlevel 1 goto :fail

echo.
echo == 4/4 Regenerate evaluation metrics (37 questions, 109 claims) ==
python evaluation\run_review2_eval.py --output-prefix demo_rerun > demo\outputs\eval_run.log
type demo\outputs\eval_run.log
if errorlevel 1 goto :fail

echo.
echo Done. Outputs: demo\outputs\ and evaluation\results\demo_rerun_report.json
exit /b 0

:fail
echo.
echo A step failed. Copy the error above and check the install steps in README.md.
exit /b 1
