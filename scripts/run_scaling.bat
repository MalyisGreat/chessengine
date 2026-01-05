@echo off
REM Quick scaling experiment launcher for Windows
REM Usage: run_scaling.bat C:\path\to\stockfish.exe

if "%1"=="" (
    echo Usage: run_scaling.bat ^<stockfish_path^>
    echo Example: run_scaling.bat C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe
    exit /b 1
)

python "%~dp0run_scaling_experiment.py" --stockfish "%1" %2 %3 %4 %5 %6 %7 %8 %9
