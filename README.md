# PhotoGuard Dockerized Workspace

This directory is a self-contained deployment bundle for PhotoGuard. It packages the React frontend, the FastAPI backend, and the production nginx proxy in one place without touching the original repo trees.

## What to hand to a teammate

The easiest onboarding path is to give them:

- this `dockerized/` folder
- access to the model directory at `../artifacts/local_inpaint_model`

Then they only need Docker.

## Structure

- `frontend/`: copied React app source from the integrated UI
- `backend/`: copied FastAPI app plus local copies of `immunization.py` and `utils.py`
- `proxy/`: nginx config plus the frontend production image build
- `build-images.sh`: builds both Docker images with plain Docker
- `run-stack.sh`: runs the full stack with plain Docker
- `run-stack-gpu.sh`: same, but adds `--gpus all` to the API container
- `stop-stack.sh`: stops the running containers
- `package-images.sh`: builds and exports both images into a tarball
- `load-images.sh`: imports the tarball on another machine
- `push-ghcr.sh`: tags and pushes both images to GitHub Container Registry
- `pull-ghcr.sh`: pulls both images back from GitHub Container Registry
- `docker-compose.yml`: optional Compose setup if Compose is available

## Easiest local flow

Build:

```bash
cd /home/tobi/photoguard/dockerized
./build-images.sh
```

Run on CPU:

```bash
cd /home/tobi/photoguard/dockerized
./run-stack.sh
```

Run with GPU:

```bash
cd /home/tobi/photoguard/dockerized
./run-stack-gpu.sh
```

Stop:

```bash
cd /home/tobi/photoguard/dockerized
./stop-stack.sh
```

Then open:

```text
http://localhost:8080
```

## Package for another machine

Create a portable image bundle:

```bash
cd /home/tobi/photoguard/dockerized
./package-images.sh
```

That creates:

```text
photoguard-images.tar
```

On the target machine:

```bash
cd /path/to/dockerized
./load-images.sh
./run-stack.sh
```

## Push to GitHub Container Registry

Build first, ideally with your repository URL so the package can be associated with the repo:

```bash
cd /home/tobi/photoguard/dockerized
OCI_SOURCE=https://github.com/YOUR_USER/YOUR_REPO ./build-images.sh
```

Create a GitHub personal access token (classic) with at least `write:packages`, then push:

```bash
cd /home/tobi/photoguard/dockerized
export CR_PAT=YOUR_GITHUB_CLASSIC_PAT
GHCR_NAMESPACE=YOUR_GITHUB_USER TAG=v1 ./push-ghcr.sh
```

That pushes:

```text
ghcr.io/YOUR_GITHUB_USER/photoguard-api:v1
ghcr.io/YOUR_GITHUB_USER/photoguard-web:v1
```

The namespace portion of the image path is forced to lowercase automatically for Docker compatibility, but your GitHub login username can still use its normal casing.

If you also want `latest` pushed at the same time:

```bash
GHCR_NAMESPACE=YOUR_GITHUB_USER TAG=v1 PUSH_LATEST=1 ./push-ghcr.sh
```

On another machine, pull them back and reuse the normal run script:

```bash
cd /path/to/dockerized
GHCR_NAMESPACE=YOUR_GITHUB_USER TAG=v1 ./pull-ghcr.sh
./run-stack.sh
```

## Model mount

By default, the run scripts mount:

```text
../artifacts/local_inpaint_model
```

to:

```text
/app/models/local_inpaint_model
```

If your model lives somewhere else, override it at runtime:

```bash
MODEL_DIR=/absolute/path/to/local_inpaint_model ./run-stack.sh
```

## Optional Compose flow

If your machine has Docker Compose available:

```bash
cd /home/tobi/photoguard/dockerized
docker compose up --build
```

GPU:

```bash
cd /home/tobi/photoguard/dockerized
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Notes

- The original `backend/`, `frontend-src/`, and `intergrated/` folders remain untouched.
- The frontend uses `/api` as its base URL, and nginx handles production proxying.
- The web container now accepts `API_UPSTREAM` at runtime, so the Azure frontend can proxy to a remote GPU backend without rebuilding the UI.
- The API container is based on `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04` and installs PyTorch from the CUDA 13 nightly index.
- GitHub’s Container registry docs for authentication and pushing are here:
  https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry

## Azure frontend with a remote GPU backend

If you want Azure to host only the frontend, point the web container at a remote backend instead of running the local API container:

```bash
docker run -d \
  --name photoguard-web \
  -p 80:80 \
  -e API_UPSTREAM=http://YOUR_GPU_BACKEND_IP:8000 \
  photoguard-web
```

If the remote backend is public and protected by TLS, use `https://...` instead.

This keeps the browser on the Azure origin while nginx proxies `/api/*` server-side to the GPU host.
