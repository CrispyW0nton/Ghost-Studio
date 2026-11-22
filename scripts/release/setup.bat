@echo off
setlocal
title Ghost Studio setup
echo Ghost Studio setup
echo ------------------
echo This installs the Python packages Ghost Studio needs into your Python 3.13.
echo It needs an internet connection and can take a few minutes.
echo.

py -3.13 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if errorlevel 1 (
    echo 64-bit Python 3.13 was not found.
    echo Install it from https://www.python.org/downloads/ ^(Windows installer, 64-bit^),
    echo keep the default install location, then run this file again.
    pause
    exit /b 1
)

py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo Package install failed. Scroll up for the error, or ask in #ghost-studio-help.
    pause
    exit /b 1
)

echo.
echo Setup finished. Start Ghost Studio with GhostStudio.exe in this folder.
pause
