# AI Foundry Frontend

This is a small Vite + React scaffold converted from the provided `index.html` UI.

Quick start (Windows bash):

```bash
cd ai-foundry-frontend
npm install
npm run dev
```

Then open the URL printed by Vite (usually http://localhost:5173) and the app will connect to the existing FastAPI websocket at `ws://localhost:8000/ws_stream_campaign`.

Notes:
- The frontend expects the Python FastAPI server from your workspace to run on port 8000.
- If your server is on a different host/port, update the WebSocket URL in `src/App.jsx`.
- Optional: use `pnpm` or `yarn` instead of `npm`.
