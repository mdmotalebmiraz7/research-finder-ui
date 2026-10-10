
#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Research Finder..."

# Start FastAPI backend
cd "$PROJECT_DIR/backend"
source venv/bin/activate
uvicorn research_finder:app --reload --port 8000 &
BACKEND_PID=$!

# Start React frontend
cd "$PROJECT_DIR/frontend"
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

# Stop both when the script is stopped
cleanup() {
    echo "Stopping Research Finder..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

wait
