#!/bin/bash
# Script to run the FastAPI backend (with reload) and then the GUI client

# Start the FastAPI API server with uvicorn in the background
cd "$(dirname "$0")"
uvicorn src.api:app --reload &
API_PID=$!

# Wait a few seconds to ensure the API is up (adjust as needed)
sleep 3

# Run the GUI client (this will block until GUI is closed)
uv run src/gui_client.py

# After GUI closes, stop the API server
kill $API_PID
