@echo off
echo Building DevController...
python -m PyInstaller --noconfirm --onefile --console --name "DevController" --add-data "config.yml;." --hidden-import requests main.py
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)
echo Build success! Executable is in dist/DevController.exe
pause
