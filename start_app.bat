@echo off
REM Script to run the FastAPI backend (with reload) and then the GUI client

REM Start the FastAPI API server with uvicorn in the background
start "" cmd /c "uvicorn src.api:app --reload"
REM Wait a few seconds to ensure the API is up (adjust as needed)
timeout /t 3 >nul

REM Run the GUI client (this will block until GUI is closed)
uv run src\gui_client.py

REM After GUI closes, kill the uvicorn process
for /f "tokens=2" %%a in ('tasklist ^| findstr uvicorn') do taskkill /PID %%a /F
