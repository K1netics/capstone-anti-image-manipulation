# PhotoGuard Workspace Audit

This repository currently contains one tracked legacy app plus several newer workspace copies for integration and deployment experiments.

## Current sections

- `backend/`
  - Original tracked Gradio-based PhotoGuard app.
  - Uses local sample images in `backend/images/`.
- `artifacts/`
  - Local model artifacts and frozen dependency snapshots.
  - Large local-only data directory; ignored by git.
- `intergrated/`
  - React frontend + FastAPI backend development workspace.
  - Now includes integrated request feedback/status logs surfaced from the API to the UI.
  - Best candidate for the main app if you want a normal frontend/backend architecture.
- `feedback.py` and `run_feedback_demo.py`
  - Shared feedback prototype for user-facing validation and confirmation messages.
  - `feedback.py` now re-exports the integrated workspace helper so the standalone demo and integrated app use the same message format.
- `dockerized/`
  - Deployment bundle with Dockerfiles, nginx proxy, Compose files, and copied app code.
  - Best candidate for packaged deployment.
- `gcollab/`
  - Backend-only FastAPI bundle for Google Colab.
- `better-pytorch-cu124/`
  - Backend-only FastAPI bundle for a remote GPU machine that already has PyTorch installed.
- `frontend-src/`
  - Earlier frontend-only prototype.
  - Talks to `http://localhost:8000/embed`, which does not match the newer `/process` API flow.
- Root `src/`, `public/`, `package.json`, `vite.config.ts`
  - Leftover Vite starter app.
  - Not the real PhotoGuard frontend.

## What looks complete

- `backend/`: legacy standalone backend UI
- `intergrated/`: frontend/backend integration workspace
- `dockerized/`: dockerized deployment workspace
- `gcollab/`: backend-only Colab workspace
- `better-pytorch-cu124/`: backend-only remote GPU workspace

## What looks unnecessary or ready to archive

- Root Vite starter files:
  - `src/`
  - `public/`
  - `index.html`
  - `package.json`
  - `package-lock.json`
  - `tsconfig*.json`
  - `vite.config.ts`
  - `eslint.config.js`
- `frontend-src/`
  - Superseded by `intergrated/`
- Root `Dockerfile`
  - Older one-off container path; `dockerized/` is more complete
- Generated or local-only directories:
  - `intergrated/node_modules/`
  - `intergrated/.venv/`
  - `__pycache__/`
  - `.gradio/`
  - `demo/.gradio/`
- Nested repos that will complicate cleanup if you want one git repo:
  - `frontend-src/.git/`
  - `intergrated/.git/`

## Recommended target structure

If you want to clean this repo up without losing work, this is the simplest direction:

```text
photoguard/
  apps/
    legacy-gradio/
    integrated-web/
    integrated-api/
  deploy/
    docker/
    colab-backend/
    remote-gpu-backend/
  artifacts/
  README.md
```

## Recommended cleanup order

1. Keep `intergrated/` as the main development app.
2. Keep `dockerized/` as the deployment package.
3. Keep `gcollab/` and `better-pytorch-cu124/` only if you still need those deployment targets.
4. Archive or delete `frontend-src/`, the root Vite starter files, and the old root `Dockerfile` if they are no longer needed.
5. Remove generated directories before committing: `node_modules`, `.venv`, `__pycache__`, `.gradio`.
6. Rename `intergrated/` to `integrated/` once you are ready to update hard-coded paths and docs.

## Verification notes

- The tracked legacy backend now parses again after fixing a syntax error in `backend/app.py`.
- The FastAPI entrypoints in `intergrated/`, `dockerized/`, `gcollab/`, and `better-pytorch-cu124/` compile successfully.
- Frontend build verification is incomplete in this shell because the local Node version is `v12.22.9`, which is too old for the current TypeScript/Vite toolchain, and some workspaces do not have dependencies installed.
