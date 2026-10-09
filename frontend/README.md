# Research Finder UI

React + Vite frontend for the Research Finder FastAPI backend.

## Features
- First-open Groq API key popup; key is saved in this browser's localStorage so it only needs to be entered once per browser.
- Demo mode removed; research, PDF processing, and paper Q&A use the live backend.
- Live backend activity panel polls `GET /logs` during research.
- Creator credit: Md Miraz Ali, Dhaka University, EEE Department.

## Run in Google Colab
1. Keep the FastAPI backend running on port 8000.
2. Start the UI:
   ```bash
   npm install
   npm run dev -- --host 0.0.0.0
   ```
3. For a temporary public link, run `cloudflared tunnel --url http://127.0.0.1:5173` in another cell and use the new `trycloudflare.com` URL.

The Vite proxy forwards `/api/*` to `http://127.0.0.1:8000/*`. `allowedHosts: true` is set for temporary Colab/Cloudflare testing only; restrict allowed hosts before production deployment.

The API key is stored in the browser's localStorage and included in requests to the backend. Do not use a shared/public computer for a personal API key.
