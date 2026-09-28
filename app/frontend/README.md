# Demo portal UI (React + Vite)

The browser side of the Sprint 1 demo portal. See the "Demo portal" section of the project
[README](../../README.md) for setup and running.

- `src/App.jsx` — page: sample picker / upload, box overlay, threshold slider, detections table, model panel
- `src/detections.js` — threshold filter, fragment merging and label matching (same logic as `ml/predict.py`)
- `npm run dev` — dev server on :5173 (API proxied to the FastAPI backend on :8000)
- `npm run build` — production build to `dist/`, served by `app/backend/main.py`
