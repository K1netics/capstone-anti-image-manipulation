from __future__ import annotations

import base64
import gc
import io
import os
import traceback
from dataclasses import replace
from pathlib import Path
from threading import Lock

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .immunization import ImmunizationConfig, immunize_image
from .utils import recover_image

DEFAULT_INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"
DEFAULT_LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "local_inpaint_model"
MAX_IMAGE_DIMENSION = 512
DIMENSION_MULTIPLE = 32
DEFAULT_SEED = 1234

DEFAULT_IMMUNIZATION_CONFIG = ImmunizationConfig(
    profile_name="stable_diffusion",
    target_mode="random",
    iters=20,
    target_strength=1.0,
    chaos_strength=0.35,
    denoiser_strength=0.2,
    denoiser_steps=1,
    eot_samples=1,
    resize_jitter=0.05,
    noise_strength=0.01,
    blur_kernel_size=3,
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


DEFAULT_MODEL_SOURCE = (
    str(DEFAULT_LOCAL_MODEL_DIR) if DEFAULT_LOCAL_MODEL_DIR.exists() else DEFAULT_INPAINT_MODEL_ID
)
INPAINT_MODEL_SOURCE = os.getenv("PHOTOGUARD_INPAINT_MODEL", DEFAULT_MODEL_SOURCE)
INPAINT_LOCAL_FILES_ONLY = _env_flag(
    "PHOTOGUARD_LOCAL_FILES_ONLY",
    DEFAULT_LOCAL_MODEL_DIR.exists(),
)
PIPELINE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DTYPE = torch.float16 if PIPELINE_DEVICE == "cuda" else torch.float32

_pipeline: StableDiffusionInpaintPipeline | None = None
_pipeline_error: str | None = None
_pipeline_init_lock = Lock()
_pipeline_run_lock = Lock()


class OutputImage(BaseModel):
    label: str
    dataUrl: str


class ProcessResponse(BaseModel):
    outputs: list[OutputImage]
    device: str
    modelSource: str
    processingMode: str


class HealthResponse(BaseModel):
    status: str
    device: str
    modelSource: str
    pipelineReady: bool


app = FastAPI(title="PhotoGuard Docker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_runtime_memory() -> None:
    gc.collect()
    if PIPELINE_DEVICE == "cuda":
        torch.cuda.empty_cache()


def _get_pipeline() -> StableDiffusionInpaintPipeline:
    global _pipeline, _pipeline_error

    if _pipeline is not None:
        return _pipeline

    with _pipeline_init_lock:
        if _pipeline is not None:
            return _pipeline

        try:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                INPAINT_MODEL_SOURCE,
                local_files_only=INPAINT_LOCAL_FILES_ONLY,
                torch_dtype=PIPELINE_DTYPE,
                safety_checker=None,
            )
            pipeline = pipeline.to(PIPELINE_DEVICE)
            if PIPELINE_DEVICE == "cpu":
                pipeline.enable_attention_slicing()
            _pipeline = pipeline
            _pipeline_error = None
        except Exception as exc:  # pragma: no cover - startup failure path
            _pipeline_error = str(exc)
            raise RuntimeError(
                "Unable to load the inpainting model. Set PHOTOGUARD_INPAINT_MODEL to a valid "
                f"local directory or repo id. Current source: {INPAINT_MODEL_SOURCE!r}."
            ) from exc

    return _pipeline


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{upload.filename}' is empty.")
    return data


def _normalize_seed(seed: str | None) -> int:
    if seed is None or not seed.strip():
        return DEFAULT_SEED

    try:
        return int(seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Seed must be an integer.") from exc


def _load_rgb_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_mask_image(data: bytes) -> Image.Image:
    raw_image = Image.open(io.BytesIO(data)).convert("RGBA")
    black_background = Image.new("RGBA", raw_image.size, (0, 0, 0, 255))
    composited = Image.alpha_composite(black_background, raw_image)
    return composited.convert("L")


def _image_to_data_url(image: Image.Image, *, format_name: str = "WEBP") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format_name.lower()};base64,{encoded}"


def _snap_dimension(value: int) -> int:
    safe_value = max(DIMENSION_MULTIPLE, int(value))
    return max(DIMENSION_MULTIPLE, (safe_value // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)


def _normalized_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width >= height:
        target_width = _snap_dimension(min(width, MAX_IMAGE_DIMENSION))
        scale = target_width / width
        target_height = _snap_dimension(height * scale)
        return target_width, target_height

    target_height = _snap_dimension(min(height, MAX_IMAGE_DIMENSION))
    scale = target_height / height
    target_width = _snap_dimension(width * scale)
    return target_width, target_height


def _prepare_request_images(image_data: bytes, mask_data: bytes) -> tuple[Image.Image, Image.Image]:
    init_image = _load_rgb_image(image_data)
    mask_image = _load_mask_image(mask_data)
    target_size = _normalized_size(init_image.size)

    if init_image.size != target_size:
        init_image = init_image.resize(target_size, resample=Image.LANCZOS)

    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, resample=Image.NEAREST)

    return init_image, ImageOps.invert(mask_image)


def _process_request(
    image_data: bytes,
    mask_data: bytes,
    prompt: str,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    immunize: bool,
) -> ProcessResponse:
    pipeline = _get_pipeline()
    torch.manual_seed(seed)

    with _pipeline_run_lock:
        try:
            init_image, mask_image = _prepare_request_images(image_data, mask_data)

            outputs: list[tuple[Image.Image, str]] = []
            immunized_image: Image.Image | None = None

            if immunize:
                immunization_config = replace(
                    DEFAULT_IMMUNIZATION_CONFIG,
                    target_size=init_image.size,
                )
                immunized_image, _ = immunize_image(
                    init_image,
                    mask_image,
                    pipeline,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    config=immunization_config,
                    seed=seed,
                )
                outputs.append((immunized_image, "Immunized Image"))

            edited_image = pipeline(
                prompt=prompt,
                image=immunized_image if immunized_image is not None else init_image,
                mask_image=mask_image,
                height=init_image.size[1],
                width=init_image.size[0],
                eta=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

            edited_image = recover_image(edited_image, init_image, mask_image)
            outputs.append(
                (
                    edited_image,
                    "Edited After Immunization" if immunize else "Edited Image (Without Immunization)",
                )
            )

            return ProcessResponse(
                outputs=[
                    OutputImage(label=label, dataUrl=_image_to_data_url(image))
                    for image, label in outputs
                ],
                device=PIPELINE_DEVICE,
                modelSource=INPAINT_MODEL_SOURCE,
                processingMode="immunize" if immunize else "edit",
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - runtime failure path
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _cleanup_runtime_memory()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _pipeline_error is None else "offline",
        device=PIPELINE_DEVICE,
        modelSource=INPAINT_MODEL_SOURCE,
        pipelineReady=_pipeline is not None,
    )


@app.post("/process", response_model=ProcessResponse)
async def process(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    seed: str = Form(str(DEFAULT_SEED)),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(100),
    immunize: bool = Form(False),
) -> ProcessResponse:
    image_bytes = await _read_upload(image)
    mask_bytes = await _read_upload(mask)
    normalized_seed = _normalize_seed(seed)
    return _process_request(
        image_data=image_bytes,
        mask_data=mask_bytes,
        prompt=prompt,
        seed=normalized_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        immunize=immunize,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
