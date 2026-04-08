# PhotoGuard Integrated Workspace

This directory is a separate project under `/home/tobi/photoguard/intergrated`.
It keeps the original PhotoGuard demo untouched and replaces the Gradio-driven shell with a React frontend plus a Python API that live entirely in this folder.

## What is here

- `src/`: standalone React frontend
- `api/`: standalone Python API entrypoint
- `api/feedback.py`: shared feedback helpers for validation, confirmation messages, and UI-friendly status logs
- `vite.config.ts`: proxies `/api` to `http://127.0.0.1:8000` during development

## Feedback flow

The integrated workspace now includes the feedback prototype from the repo root and uses it in the real `/process` API flow.

- The backend validates uploaded source and mask images with user-friendly status messages.
- Successful requests return a `statusText` log alongside the generated images.
- Failed requests return the same feedback log in the error response so the UI can show what went wrong.
- The frontend output panel includes a "Processing feedback" block that shows these validation and confirmation messages after each run.
- The API also exposes `/progress/{request_id}` so the frontend can poll live request progress and show immunization iteration metrics while a request is still running.

## Quality and profile controls

The integrated app now exposes a few backend-facing controls in the editor:

- `Immunization profile`: `stable_diffusion` or `nano_banana_experimental`
- `Working resolution`: `512`, `1024`, or `original`
- `Output format`: `PNG` or `WebP`
- `Lossless output`: applies when `WebP` is selected

The backend composites edited outputs back onto the original full-resolution image, so untouched regions keep their source quality even when the model runs at a smaller working size.

## Run locally

1. Start the API:
   `cd /home/tobi/photoguard/intergrated && python -m uvicorn api.main:app --reload --port 8000`
2. Start the frontend:
   `cd /home/tobi/photoguard/intergrated && npm install && npm run dev`

For CUDA-heavy local sessions, prefer running the API without `--reload`:
`cd /home/tobi/photoguard/intergrated && npm run dev:api:gpu`

## Notes

- The original `backend/` code is not modified by this workspace.
- The integrated API reuses the original immunization utilities by importing them, but all new entrypoints and UI files live here.
- By default the API looks for the local model at `/home/tobi/photoguard/artifacts/local_inpaint_model`.
- The root `feedback.py` and `run_feedback_demo.py` remain available as a lightweight standalone demo of the same feedback pattern.
