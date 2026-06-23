@echo off
call conda activate AI
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0
uvicorn web.main:app --host 127.0.0.1 --port 8000
pause
