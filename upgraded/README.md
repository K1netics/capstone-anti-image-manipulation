# PhotoGuard Upgraded Workspace

This directory is a separate project under `/home/tobi/photoguard/upgraded`.
It keeps the earlier capstone app untouched and provides a second React + Python API workspace for the upgraded defense experiments.

## What is here

- `src/`: standalone React frontend
- `api/`: standalone Python API entrypoint
- `backend/`: local copy of the backend immunization code for the upgraded model work
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
   `cd /home/tobi/photoguard/upgraded && python -m uvicorn api.main:app --reload --port 8000`
2. Start the frontend with the bundled local Node 22 runtime:
   `cd /home/tobi/photoguard/upgraded && ./run-frontend.sh`

For CUDA-heavy local sessions, prefer running the API without `--reload`:
`cd /home/tobi/photoguard/upgraded && npm run dev:api:gpu`

To run npm commands with the bundled local Node 22 runtime:
`cd /home/tobi/photoguard/upgraded && ./npmw install`
`cd /home/tobi/photoguard/upgraded && ./npmw run build`

## Container images

The upgraded workspace now includes a container bundle that mirrors the old `dockerized/` workflow while using the upgraded frontend/API/backend code.

- `docker/backend.Dockerfile`: GPU-ready FastAPI image
- `docker/proxy.Dockerfile`: React build + nginx proxy image
- `docker-compose.yml`: local stack with mounted model dir and shared cache
- `docker-compose.gpu.yml`: adds `gpus: all` for the API service
- `build-images.sh`: builds local API + web images
- `push-ghcr.sh`: tags and pushes both images to GitHub Container Registry

Build locally:
`cd /home/tobi/photoguard/upgraded && ./build-images.sh`

Run with Compose:
`cd /home/tobi/photoguard/upgraded && docker compose up --build`

Run with GPU access:
`cd /home/tobi/photoguard/upgraded && docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`

Push to GHCR:
`cd /home/tobi/photoguard/upgraded && GHCR_NAMESPACE=YOUR_GITHUB_USER TAG=v1 ./push-ghcr.sh`

The API image defaults to:
- `PHOTOGUARD_REQUIRE_FULL_STRENGTH=1`
- `PHOTOGUARD_ENABLE_XFORMERS=1`
- `PHOTOGUARD_LOCAL_FILES_ONLY=0`

The web proxy image supports three backend routes:
- `/api/`: default backend from `API_UPSTREAM`
- `/api/2.3.1/`: alternate backend from `API_UPSTREAM_231`
- `/api/2.3.2/`: alternate backend from `API_UPSTREAM_232`

That makes it easy to run one Azure frontend against two backend versions at once. The upgraded frontend also understands:
- `?backend=2.3.1`
- `?backend=2.3.2`

So you can A/B test from the same web deployment by opening:
- `https://YOUR_HOST/`
- `https://YOUR_HOST/?backend=2.3.1`
- `https://YOUR_HOST/?backend=2.3.2`

That means the container can use a mounted local model directory at `/app/models/local_inpaint_model`, or fall back to Hugging Face download/cache when no local model files are present.

## Notes

- The earlier capstone app is not modified by this workspace.
- This upgraded API resolves backend imports only from `upgraded/backend`, so you can change the defense logic here without affecting the other app.
- A placeholder local model directory is included at `/home/tobi/photoguard/upgraded/artifacts/local_inpaint_model`.
- The upgraded app only auto-uses that directory when it contains a valid Diffusers model with `model_index.json`; otherwise it falls back to the configured repo id.
- Set `PHOTOGUARD_COMPILE=1` to opt into `torch.compile` for the upgraded API/standalone backend.
- `upgraded` defaults to full-strength, fail-closed immunization for honest evaluation. If a run runs out of GPU memory, it errors instead of silently weakening the defense.
- Set `PHOTOGUARD_REQUIRE_FULL_STRENGTH=0` if you explicitly want to re-enable the lower-memory fallback path.
- The root `feedback.py` and `run_feedback_demo.py` remain available as a lightweight standalone demo of the same feedback pattern.

## Training stack

The upgraded workspace now includes an SDXL teacher-export and student-protector training path:

- `scripts/generate_teacher_set.py`: exports protected teacher images and deltas
- `scripts/train_protector.py`: trains a single-pass protector model from that teacher data
- `scripts/eval_external_edits.py`: creates external-eval templates and scores DeeVid/NB2 results after you fill in the edited output paths
- `training-requirements.txt`: optional extra dependencies for training and external evaluation

Install the extra training dependencies into your training environment:

`cd /home/tobi/photoguard/upgraded && pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0`
`cd /home/tobi/photoguard/upgraded && grep -Ev '^(torch|torchvision|xformers)($|[<>=!~])' api/requirements.txt > /tmp/api-reqs-no-torch.txt && pip install -r /tmp/api-reqs-no-torch.txt`
`cd /home/tobi/photoguard/upgraded && pip install -r training-requirements.txt`

Optional real face-ID branch:

`cd /home/tobi/photoguard/upgraded && pip install --no-deps facenet-pytorch`

Recommended cache setup before the first FaceNet-backed run:

`export TORCH_HOME=/workspace/.cache/torch`

Quick verification:

`cd /home/tobi/photoguard/upgraded && python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2').eval(); print('facenet ready')"`

Enable the real face-ID losses during training:

`cd /home/tobi/photoguard/upgraded && python scripts/train_protector.py --manifest /path/to/teacher/manifest.jsonl --output-dir /path/to/run --face-id-backend facenet --real-face-id-weight 0.12 --real-face-id-drift-weight 0.06`

Do not run a blind `pip install xformers` in the RunPod training venv. It can replace the working `torch==2.8.0+cu128` stack with incompatible CUDA 13 wheels.

Notable training features now included:

- early-step weighted denoiser supervision
- mask augmentation inside teacher generation
- grouped prompt-family sampling for prompt-agnostic training
- portrait face branch with both heuristic face descriptors and optional real FaceNet embeddings
- purification-aware consistency losses
- selective perturbation weighting for better visual fidelity
- optional Glaze/Mist-inspired style-cloak statistics in the backend immunizer
- optional ArtShield-style multi-view ownership watermark alignment plus the `artist_cloak` research profile

Generate a teacher set:

`cd /home/tobi/photoguard/upgraded && python scripts/generate_teacher_set.py --metadata /path/to/metadata.jsonl --output-dir /path/to/teacher`

Train a student protector:

`cd /home/tobi/photoguard/upgraded && python scripts/train_protector.py --manifest /path/to/teacher/manifest.jsonl --output-dir /path/to/run`

Create an external-eval template from teacher samples:

`cd /home/tobi/photoguard/upgraded && python scripts/eval_external_edits.py --teacher-manifest /path/to/teacher/manifest.jsonl --template-out /path/to/eval_cases.csv`

After manually filling `unprotected_edit` and `protected_edit` paths for DeeVid/NB2 outputs, score the batch:

`cd /home/tobi/photoguard/upgraded && python scripts/eval_external_edits.py --cases-csv /path/to/eval_cases.csv --report-out /path/to/eval_report.csv`
