# Immuna Frontend + Upgraded Backend Wrapper

This directory is the presentation-facing integration layer:

- frontend design comes from the team `immuna` repo
- backend contract comes from `/home/tobi/photoguard/upgraded`
- the delta is kept as a small overlay, not a forked frontend rewrite

## What changed

The previous adapter path was the wrong abstraction for the upgraded backend. The stable path is now:

- sync upstream `immuna`
- apply a maintained frontend overlay
- talk directly to the upgraded backend
- use same-origin `/api` proxying in Docker to avoid CORS and mixed-content issues

## Files that matter

- `sync-frontend-upgraded.sh`
  - one entrypoint to sync upstream frontend code and re-apply the backend-compat overlay
- `frontend-overlay-upgraded/`
  - minimal maintained delta over the team frontend
- `web.Dockerfile`
  - builds the synced frontend and serves it behind nginx with `/api` proxying
- `docker-compose.yml`
  - runs the synced Immuna frontend against the upgraded backend directly

## Local-first workflow

Use this while stabilizing the backend locally.

1. Sync the latest team frontend and apply the overlay:

`./sync-frontend-upgraded.sh /path/to/local/immuna`

Or directly from GitHub:

`./sync-frontend-upgraded.sh https://github.com/mochibearr/immuna.git`

2. Start the upgraded backend locally:

`cd /home/tobi/photoguard/upgraded && python -m uvicorn api.main:app --reload --port 8000`

3. Start the synced frontend locally:

`cd /home/tobi/photoguard/immuna-meeting-docker/frontend && npm ci && npm run dev`

The overlay supplies a Vite `/api` proxy to `http://127.0.0.1:8000`, so local frontend development stays same-origin and does not depend on manual API URL edits.

## Docker workflow

Before building, sync the frontend once:

`cd /home/tobi/photoguard/immuna-meeting-docker && ./sync-frontend-upgraded.sh /path/to/local/immuna`

Build images:

`cd /home/tobi/photoguard/immuna-meeting-docker && ./build-images.sh`

Run the stack:

`cd /home/tobi/photoguard/immuna-meeting-docker && docker compose up --build`

Open:

- frontend: `http://localhost:8080`
- backend health: `http://localhost:8000/health`

## Runtime config

- frontend default runtime API base: `/api`
- nginx proxies `/api/*` to the upgraded backend container
- override upstream target only if you intentionally want the web image to point somewhere else:
  - `API_UPSTREAM=http://HOST:PORT`

## Notes

- This wrapper is for presentation integration, not research training.
- Keep backend experimentation in `/home/tobi/photoguard/upgraded` until the API path is stable.
- Only move back to a remote GPU box when local VRAM is actually insufficient.
