@echo off
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONLEGACYWINDOWSSTDIO=0
set PYTHONIOENCODING=utf-8
".\.venv\Scripts\python.exe" main.py
pause