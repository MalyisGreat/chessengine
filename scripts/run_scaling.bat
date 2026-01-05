@echo off
REM Quick scaling experiment launcher for Windows
REM Usage: run_scaling.bat [optional: stockfish_path] [extra args]
REM Without arguments, auto-downloads Stockfish 16.1

if "%1"=="" (
    echo Auto-downloading Stockfish 16.1 and running scaling experiment...
    python "%~dp0run_scaling_experiment.py" --auto-stockfish
) else (
    python "%~dp0run_scaling_experiment.py" --stockfish "%1" %2 %3 %4 %5 %6 %7 %8 %9
)
