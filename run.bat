@echo off
setlocal
cd /d "%~dp0"

REM Default mode is infer. You can pass train|infer|both as the first arg:
REM   run.bat train
REM   run.bat infer
REM   run.bat both

set MODE=%1
if "%MODE%"=="" set MODE=infer

powershell -NoProfile -ExecutionPolicy Bypass -File ".\run.ps1" -Mode %MODE%
endlocal