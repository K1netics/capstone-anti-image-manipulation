# PhotoGuard Integrated Workspace

This directory is a separate project under `/home/tobi/photoguard/intergrated`.
It keeps the original PhotoGuard demo untouched and replaces the Gradio-driven shell with a React frontend plus a Python API that live entirely in this folder.

## What is here

- `src/`: standalone React frontend
- `api/`: standalone Python API entrypoint
- `vite.config.ts`: proxies `/api` to `http://127.0.0.1:8000` during development

## Run locally

1. Start the API:
   `cd /home/tobi/photoguard/intergrated && python -m uvicorn api.main:app --reload --port 8000`
2. Start the frontend:
   `cd /home/tobi/photoguard/intergrated && npm install && npm run dev`

## Notes

- The original `backend/` code is not modified by this workspace.
- The integrated API reuses the original immunization utilities by importing them, but all new entrypoints and UI files live here.
- By default the API looks for the local model at `/home/tobi/photoguard/artifacts/local_inpaint_model`.
