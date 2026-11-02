@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" goto missing_python
".venv\Scripts\python.exe" -X utf8 -m examples.demo_ar %*
set "arExit=%errorlevel%"
if not "%arExit%"=="0" pause
popd
exit /b %arExit%
:missing_python
echo Project Python was not found: .venv\Scripts\python.exe
pause
popd
exit /b 1
