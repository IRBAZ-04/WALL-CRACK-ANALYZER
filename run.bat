@echo off
title Automatic Wall Crack Analyzer
cls
echo ==============================================================
echo        AUTOMATIC WALL CRACK ANALYZER (UCSC313L PROJECT)
echo           100%% From-Scratch Algorithms - Zero OpenCV
echo ==============================================================
echo.
echo Launching PySide6 Desktop GUI...
echo.

.\.venv\Scripts\python.exe gui\app_pyside.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Info] Launching classic Tkinter GUI fallback...
    .\.venv\Scripts\python.exe gui\app.py
)

echo.
echo Application closed.
pause
