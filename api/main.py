from __future__ import annotations

import base64
import gc
import io
import os
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from uuid import uuid4

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

CURRENT_FILE = Path(__file__).resolve()


def _configured_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _candidate_workspace_roots() -> list[Path]:
    configured_root = _configured_path("PHOTOGUARD_WORKSPACE_ROOT")
    candidates: list[Path] = []
    if configured_root is not None:
        candidates.append(configured_root)
    candidates.extend(CURRENT_FILE.parents)
    return candidates


def _find_original_backend_root() -> Path:
    configured_backend_root = _configured_path("PHOTOGUARD_BACKEND_ROOT")
    if configured_backend_root is not None:
        return configured_backend_root

    for root in _candidate_workspace_roots():
        candidate = root / "backend"
        if (candidate / "immunization.py").exists() and (candidate / "utils.py").exists():
            return candidate

    return CURRENT_FILE.parents[2] / "backend"


def _default_local_model_dir() -> Path:
    configured_model_dir = _configured_path("PHOTOGUARD_MODEL_DIR")
    if configured_model_dir is not None:
        return configured_model_dir

    for root in _candidate_workspace_roots():
        candidate = root / "artifacts" / "local_inpaint_model"
        if candidate.exists():
            return candidate

    return _find_original_backend_root().parent / "artifacts" / "local_inpaint_model"


ORIGINAL_BACKEND_ROOT = _find_original_backend_root()
PROJECT_ROOT = ORIGINAL_BACKEND_ROOT.parent
if str(ORIGINAL_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_BACKEND_ROOT))

from immunization import (  # noqa: E402
    ImmunizationConfig,
    get_available_immunization_profiles,
    immunize_image,
)
from utils import recover_image  # noqa: E402

try:  # pragma: no cover - import path depends on how the module is launched
    from .feedback import Feedback, validate_image_bytes
except ImportError:  # pragma: no cover
    from feedback import Feedback, validate_image_bytes

DEFAULT_INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"
DEFAULT_LOCAL_MODEL_DIR = _default_local_model_dir()
DIMENSION_MULTIPLE = 32
DEFAULT_SEED = 1234
SUPPORTED_WORKING_RESOLUTIONS = ("512", "1024", "original")
SUPPORTED_OUTPUT_FORMATS = ("png", "webp")
SUPPORTED_IMMUNIZATION_PROFILES = get_available_immunization_profiles()
DELEGATED_IMMUNIZATION_PROFILES = {
    "nano_banana_experimental": "stable_diffusion",
}

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
DEFAULT_IMMUNIZATION_PROFILE = os.getenv(
    "PHOTOGUARD_DEFAULT_IMMUNIZATION_PROFILE",
    DEFAULT_IMMUNIZATION_CONFIG.profile_name,
).strip().lower() or DEFAULT_IMMUNIZATION_CONFIG.profile_name
if DEFAULT_IMMUNIZATION_PROFILE not in SUPPORTED_IMMUNIZATION_PROFILES:
    DEFAULT_IMMUNIZATION_PROFILE = DEFAULT_IMMUNIZATION_CONFIG.profile_name
DEFAULT_WORKING_RESOLUTION = os.getenv("PHOTOGUARD_DEFAULT_WORKING_RESOLUTION", "1024").strip().lower() or "1024"
if DEFAULT_WORKING_RESOLUTION not in SUPPORTED_WORKING_RESOLUTIONS:
    DEFAULT_WORKING_RESOLUTION = "1024"
DEFAULT_OUTPUT_FORMAT = os.getenv("PHOTOGUARD_OUTPUT_FORMAT", "png").strip().lower() or "png"
if DEFAULT_OUTPUT_FORMAT not in SUPPORTED_OUTPUT_FORMATS:
    DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_IMMUNIZATION_RESOLUTION = 512
IMMUNIZATION_FALLBACK_RESOLUTION = 384


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
DEFAULT_OUTPUT_LOSSLESS = _env_flag("PHOTOGUARD_OUTPUT_LOSSLESS", True)
PIPELINE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DTYPE = torch.float16 if PIPELINE_DEVICE == "cuda" else torch.float32

_pipeline: StableDiffusionInpaintPipeline | None = None
_pipeline_error: str | None = None
_pipeline_init_lock = Lock()
_pipeline_run_lock = Lock()
_request_progress_lock = Lock()
_request_progress: dict[str, dict[str, object]] = {}


class OutputImage(BaseModel):
    label: str
    dataUrl: str


class ProcessResponse(BaseModel):
    requestId: str
    outputs: list[OutputImage]
    device: str
    modelSource: str
    processingMode: str
    statusText: str
    immunizationProfile: str
    workingResolution: str
    outputFormat: str
    losslessOutput: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    modelSource: str
    pipelineReady: bool


class ProcessProgressResponse(BaseModel):
    requestId: str
    status: str
    stage: str
    percent: float
    message: str | None = None
    iteration: int | None = None
    totalIterations: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    statusText: str


@dataclass(frozen=True)
class PreparedRequestImages:
    original_image: Image.Image
    original_edit_mask: Image.Image
    working_image: Image.Image
    working_edit_mask: Image.Image
    working_crop_box: tuple[int, int, int, int]


app = FastAPI(title="PhotoGuard Integrated API")
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
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _configure_pipeline_memory(pipeline: StableDiffusionInpaintPipeline) -> None:
    xformers_enabled = False
    if PIPELINE_DEVICE == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
        except Exception:
            xformers_enabled = False
        try:
            pipeline.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipeline.vae.enable_tiling()
        except Exception:
            pass

    if PIPELINE_DEVICE == "cpu" or not xformers_enabled:
        pipeline.enable_attention_slicing()


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _normalize_request_id(request_id: str | None) -> str:
    normalized = (request_id or "").strip()
    return normalized or f"pg-{uuid4().hex}"


def _set_request_progress(
    request_id: str,
    *,
    status: str,
    stage: str,
    percent: float,
    status_text: str,
    message: str | None = None,
    iteration: int | None = None,
    total_iterations: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    snapshot = {
        "requestId": request_id,
        "status": status,
        "stage": stage,
        "percent": round(max(0.0, min(100.0, percent)), 1),
        "message": message,
        "iteration": iteration,
        "totalIterations": total_iterations,
        "metrics": metrics or {},
        "statusText": status_text,
    }
    with _request_progress_lock:
        _request_progress[request_id] = snapshot


def _get_request_progress(request_id: str) -> ProcessProgressResponse:
    with _request_progress_lock:
        snapshot = _request_progress.get(request_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Unknown request id.")
    return ProcessProgressResponse(**snapshot)


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
            _configure_pipeline_memory(pipeline)
            _pipeline = pipeline
            _pipeline_error = None
        except Exception as exc:  # pragma: no cover - startup failure path
            _pipeline_error = str(exc)
            raise RuntimeError(
                "Unable to load the inpainting model. Set PHOTOGUARD_INPAINT_MODEL to a valid "
                f"local directory or repo id. Current source: {INPAINT_MODEL_SOURCE!r}."
            ) from exc

    return _pipeline


async def _read_upload(upload: UploadFile, fb: Feedback, *, label: str) -> bytes:
    data = await upload.read()
    try:
        validate_image_bytes(data, upload.filename, fb, label=label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=fb.get_status_text()) from exc
    return data


def _normalize_seed(seed: str | None, fb: Feedback) -> int:
    if seed is None or not seed.strip():
        fb.info(f"No seed provided. Falling back to default seed {DEFAULT_SEED}.")
        return DEFAULT_SEED

    try:
        normalized_seed = int(seed)
    except ValueError as exc:
        fb.error("Seed must be an integer.")
        raise HTTPException(status_code=400, detail=fb.get_status_text()) from exc

    fb.info(f"Using seed {normalized_seed}.")
    return normalized_seed


def _normalize_immunization_profile(profile_name: str | None, fb: Feedback) -> str:
    normalized = (profile_name or DEFAULT_IMMUNIZATION_PROFILE).strip().lower()
    if normalized not in SUPPORTED_IMMUNIZATION_PROFILES:
        available = ", ".join(SUPPORTED_IMMUNIZATION_PROFILES)
        fb.error(f"Unsupported immunization profile. Available profiles: {available}.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    fb.info(f"Using immunization profile '{normalized}'.")
    return normalized


def _resolve_immunization_profile(profile_name: str, fb: Feedback) -> str:
    delegated_profile = DELEGATED_IMMUNIZATION_PROFILES.get(profile_name)
    if delegated_profile is None:
        return profile_name

    fb.info(
        "Nano Banana experimental is temporarily delegated to the Stable Diffusion defense "
        "while we isolate profile-specific corruption."
    )
    return delegated_profile


def _normalize_working_resolution(working_resolution: str | None, fb: Feedback) -> str:
    normalized = (working_resolution or DEFAULT_WORKING_RESOLUTION).strip().lower()
    if normalized not in SUPPORTED_WORKING_RESOLUTIONS:
        fb.error("Working resolution must be one of 512, 1024, or original.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    fb.info(f"Working resolution mode: {normalized}.")
    return normalized


def _normalize_output_format(output_format: str | None, fb: Feedback) -> str:
    normalized = (output_format or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        fb.error("Output format must be 'png' or 'webp'.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    return normalized


def _load_rgb_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_mask_image(data: bytes) -> Image.Image:
    raw_image = Image.open(io.BytesIO(data)).convert("RGBA")
    black_background = Image.new("RGBA", raw_image.size, (0, 0, 0, 255))
    composited = Image.alpha_composite(black_background, raw_image)
    return composited.convert("L")


def _image_to_data_url(
    image: Image.Image,
    *,
    output_format: str,
    lossless_output: bool,
) -> str:
    buffer = io.BytesIO()
    if output_format == "png":
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    elif output_format == "webp":
        image.save(
            buffer,
            format="WEBP",
            quality=92,
            lossless=lossless_output,
            method=6,
        )
        mime = "image/webp"
    else:  # pragma: no cover - normalized earlier
        raise ValueError(f"Unsupported output format: {output_format}")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _snap_dimension(value: int) -> int:
    safe_value = max(DIMENSION_MULTIPLE, int(value))
    return max(DIMENSION_MULTIPLE, (safe_value // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)


def _snap_up_dimension(value: int) -> int:
    safe_value = max(DIMENSION_MULTIPLE, int(value))
    return max(DIMENSION_MULTIPLE, ((safe_value + DIMENSION_MULTIPLE - 1) // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)


def _normalized_size(size: tuple[int, int], max_image_dimension: int) -> tuple[int, int]:
    width, height = size
    if width >= height:
        target_width = _snap_dimension(min(width, max_image_dimension))
        scale = target_width / width
        target_height = _snap_dimension(height * scale)
        return target_width, target_height

    target_height = _snap_dimension(min(height, max_image_dimension))
    scale = target_height / height
    target_width = _snap_dimension(width * scale)
    return target_width, target_height


def _pad_image_to_canvas(image: Image.Image, target_size: tuple[int, int], *, fill: int | tuple[int, int, int]) -> Image.Image:
    if image.size == target_size:
        return image

    canvas = Image.new(image.mode, target_size, color=fill)
    canvas.paste(image, (0, 0))
    return canvas


def _prepare_request_images(
    image_data: bytes,
    mask_data: bytes,
    *,
    working_resolution: str,
    fb: Feedback | None = None,
) -> PreparedRequestImages:
    original_image = _load_rgb_image(image_data)
    protected_mask = _load_mask_image(mask_data)
    if protected_mask.size != original_image.size:
        if fb is not None:
            fb.info(
                "Resizing the painted protection mask to match the original upload before processing."
            )
        protected_mask = protected_mask.resize(original_image.size, resample=Image.NEAREST)
    original_edit_mask = ImageOps.invert(protected_mask)

    visible_image = original_image
    visible_edit_mask = original_edit_mask

    if working_resolution != "original":
        max_image_dimension = int(working_resolution)
        if max(original_image.size) > max_image_dimension:
            target_size = _normalized_size(original_image.size, max_image_dimension)
            visible_image = original_image.resize(target_size, resample=Image.LANCZOS)
            visible_edit_mask = original_edit_mask.resize(target_size, resample=Image.NEAREST)

    canvas_size = (
        _snap_up_dimension(visible_image.size[0]),
        _snap_up_dimension(visible_image.size[1]),
    )
    edge_fill = visible_image.getpixel((visible_image.size[0] - 1, visible_image.size[1] - 1))
    working_image = _pad_image_to_canvas(visible_image, canvas_size, fill=edge_fill)
    working_edit_mask = _pad_image_to_canvas(visible_edit_mask, canvas_size, fill=0)

    return PreparedRequestImages(
        original_image=original_image,
        original_edit_mask=original_edit_mask,
        working_image=working_image,
        working_edit_mask=working_edit_mask,
        working_crop_box=(0, 0, visible_image.size[0], visible_image.size[1]),
    )


def _finalize_output_image(
    image: Image.Image,
    prepared_images: PreparedRequestImages,
    *,
    background: bool = False,
) -> Image.Image:
    cropped_image = image.crop(prepared_images.working_crop_box)
    if cropped_image.size != prepared_images.original_image.size:
        cropped_image = cropped_image.resize(prepared_images.original_image.size, resample=Image.LANCZOS)

    return recover_image(
        cropped_image,
        prepared_images.original_image,
        prepared_images.original_edit_mask,
        background=background,
    )


def _immunization_config_for_profile(profile_name: str, target_size: tuple[int, int]) -> ImmunizationConfig:
    config = replace(
        DEFAULT_IMMUNIZATION_CONFIG,
        profile_name=profile_name,
        target_size=target_size,
    )
    if profile_name == "nano_banana_experimental":
        return replace(
            config,
            eps=0.06,
            step_size=0.005,
            target_mode="shifted_input",
            iters=10,
            target_strength=0.45,
            chaos_strength=0.12,
            denoiser_strength=0.12,
            denoiser_steps=1,
            eot_samples=1,
            resize_jitter=0.03,
            noise_strength=0.004,
            blur_kernel_size=3,
            semantic_boundary_strength=0.06,
            semantic_ring_width=8,
            max_prompt_variants=2,
        )
    return config


def _low_memory_immunization_config(config: ImmunizationConfig) -> ImmunizationConfig:
    return replace(
        config,
        iters=min(config.iters, 10),
        denoiser_strength=0.0,
        denoiser_steps=0,
        eot_samples=1,
        resize_jitter=min(config.resize_jitter, 0.03),
        noise_strength=min(config.noise_strength, 0.005),
        blur_kernel_size=1,
        semantic_ring_width=min(config.semantic_ring_width, 8),
        max_prompt_variants=1 if config.profile_name == "nano_banana_experimental" else max(1, config.max_prompt_variants),
    )


def _resize_for_immunization_fallback(
    image: Image.Image,
    mask: Image.Image,
    *,
    max_dimension: int,
) -> tuple[Image.Image, Image.Image]:
    if max(image.size) <= max_dimension:
        return image, mask

    target_size = _normalized_size(image.size, max_dimension)
    return (
        image.resize(target_size, resample=Image.LANCZOS),
        mask.resize(target_size, resample=Image.NEAREST),
    )


def _try_immunize_once(
    image: Image.Image,
    mask: Image.Image,
    pipeline: StableDiffusionInpaintPipeline,
    *,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    config: ImmunizationConfig,
    seed: int,
    progress_callback,
) -> Image.Image | None:
    try:
        immunized_image, _ = immunize_image(
            image,
            mask,
            pipeline,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            config=config,
            seed=seed,
            progress_callback=progress_callback,
        )
        return immunized_image
    except RuntimeError as exc:
        if not _is_out_of_memory_error(exc):
            raise
        _cleanup_runtime_memory()
        return None


def _process_request(
    request_id: str,
    image_data: bytes,
    mask_data: bytes,
    prompt: str,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    immunize: bool,
    immunization_profile: str,
    working_resolution: str,
    output_format: str,
    lossless_output: bool,
    fb: Feedback,
) -> ProcessResponse:
    with _pipeline_run_lock:
        current_stage = "queued"
        current_percent = 12.0

        def report_progress(
            stage: str,
            percent: float,
            message: str,
            *,
            status: str = "running",
            iteration: int | None = None,
            total_iterations: int | None = None,
            metrics: dict[str, float] | None = None,
        ) -> None:
            nonlocal current_stage, current_percent
            current_stage = stage
            current_percent = percent
            _set_request_progress(
                request_id,
                status=status,
                stage=stage,
                percent=percent,
                message=message,
                iteration=iteration,
                total_iterations=total_iterations,
                metrics=metrics,
                status_text=fb.get_status_text(),
            )

        def immunization_progress_callback(update: dict[str, object]) -> None:
            iteration = int(update.get("iteration", 0) or 0)
            total_iterations = int(update.get("totalIterations", 0) or 0)
            ratio = (iteration / total_iterations) if total_iterations else 0.0
            metric_payload = dict(update.get("metrics", {})) if isinstance(update.get("metrics"), dict) else {}
            if isinstance(update.get("loss"), (int, float)):
                metric_payload["loss"] = float(update["loss"])
            if isinstance(update.get("stepSize"), (int, float)):
                metric_payload["stepSize"] = float(update["stepSize"])
            report_progress(
                "immunizing",
                38.0 + (42.0 * ratio),
                f"Immunization iteration {iteration}/{total_iterations}.",
                iteration=iteration,
                total_iterations=total_iterations,
                metrics=metric_payload,
            )

        try:
            fb.processing("Preparing the inpainting pipeline.")
            report_progress("loading_pipeline", 18.0, "Preparing the inpainting pipeline.")
            pipeline = _get_pipeline()
            fb.success("Inpainting pipeline ready.")
            report_progress("loading_pipeline", 28.0, "Inpainting pipeline ready.")

            torch.manual_seed(seed)

            fb.processing("Preparing the source image and mask.")
            report_progress("preparing_inputs", 32.0, "Preparing the source image and mask.")
            prepared_images = _prepare_request_images(
                image_data,
                mask_data,
                working_resolution=working_resolution,
                fb=fb,
            )
            init_image = prepared_images.working_image
            mask_image = prepared_images.working_edit_mask
            fb.success(
                "Working canvas set to "
                f"{init_image.size[0]}x{init_image.size[1]} pixels; "
                f"final compositing returns to {prepared_images.original_image.size[0]}x"
                f"{prepared_images.original_image.size[1]}."
            )
            if output_format == "png":
                fb.info("Encoding outputs as PNG.")
            else:
                fb.info(f"Encoding outputs as WebP ({'lossless' if lossless_output else 'lossy'}).")
            report_progress(
                "preparing_inputs",
                36.0,
                f"Working canvas ready at {init_image.size[0]}x{init_image.size[1]} pixels.",
            )

            outputs: list[tuple[Image.Image, str]] = []
            immunized_image: Image.Image | None = None

            if immunize:
                effective_immunization_profile = _resolve_immunization_profile(immunization_profile, fb)
                fb.processing("Applying PhotoGuard immunization before editing.")
                report_progress("immunizing", 38.0, "Applying PhotoGuard immunization before editing.")
                immunization_image, immunization_mask = _resize_for_immunization_fallback(
                    init_image,
                    mask_image,
                    max_dimension=DEFAULT_IMMUNIZATION_RESOLUTION,
                )
                if immunization_image.size != init_image.size:
                    fb.info(
                        "Using a separate 512px defense canvas for immunization to reduce runtime and GPU memory use."
                    )
                immunization_config = _immunization_config_for_profile(
                    effective_immunization_profile,
                    immunization_image.size,
                )
                immunized_working_image = _try_immunize_once(
                    immunization_image,
                    immunization_mask,
                    pipeline,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    config=immunization_config,
                    seed=seed,
                    progress_callback=immunization_progress_callback,
                )
                if immunized_working_image is None:
                    fb.warning(
                        "GPU memory ran out during immunization. Retrying with a lower-memory defense profile."
                    )
                    report_progress(
                        "immunizing",
                        current_percent,
                        "Retrying immunization with lower-memory settings.",
                    )
                    fallback_config = _low_memory_immunization_config(immunization_config)
                    immunized_working_image = _try_immunize_once(
                        immunization_image,
                        immunization_mask,
                        pipeline,
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        config=fallback_config,
                        seed=seed,
                        progress_callback=immunization_progress_callback,
                    )
                    if immunized_working_image is not None:
                        fb.warning("Low-memory immunization fallback was used. Defense strength may be reduced.")
                if immunized_working_image is None:
                    fallback_image, fallback_mask = _resize_for_immunization_fallback(
                        immunization_image,
                        immunization_mask,
                        max_dimension=IMMUNIZATION_FALLBACK_RESOLUTION,
                    )
                    if fallback_image.size != immunization_image.size:
                        fb.warning(
                            "GPU memory is still tight. Retrying immunization on an even smaller defense canvas."
                        )
                        report_progress(
                            "immunizing",
                            current_percent,
                            "Retrying immunization on an even smaller defense canvas.",
                        )
                        reduced_config = replace(
                            _low_memory_immunization_config(immunization_config),
                            target_size=fallback_image.size,
                        )
                        reduced_immunized_image = _try_immunize_once(
                            fallback_image,
                            fallback_mask,
                            pipeline,
                            prompt=prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            config=reduced_config,
                            seed=seed,
                            progress_callback=immunization_progress_callback,
                        )
                        if reduced_immunized_image is not None:
                            immunized_working_image = reduced_immunized_image
                            fb.warning(
                                "Reduced-resolution immunization fallback was used. Defense strength may be reduced."
                            )
                if immunized_working_image is None:
                    raise torch.OutOfMemoryError("Immunization exhausted all fallback attempts.")
                if immunized_working_image.size != init_image.size:
                    immunized_working_image = immunized_working_image.resize(
                        init_image.size,
                        resample=Image.LANCZOS,
                    )
                immunized_image = _finalize_output_image(
                    immunized_working_image,
                    prepared_images,
                    background=True,
                )
                outputs.append((immunized_image, "Immunized Image"))
                fb.success("Immunized image generated.")
                report_progress("immunizing", 82.0, "Immunized image generated.")
            else:
                effective_immunization_profile = immunization_profile

            fb.processing("Running the edit pass.")
            report_progress("editing", 88.0, "Running the edit pass.")
            edited_image = pipeline(
                prompt=prompt,
                image=immunized_working_image if immunize else init_image,
                mask_image=mask_image,
                height=init_image.size[1],
                width=init_image.size[0],
                eta=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

            fb.processing("Restoring the untouched regions from the original image.")
            report_progress("compositing", 95.0, "Restoring untouched regions from the original image.")
            edited_image = _finalize_output_image(edited_image, prepared_images)
            outputs.append(
                (
                    edited_image,
                    "Edited After Immunization" if immunize else "Edited Image (Without Immunization)",
                )
            )
            fb.success("Edited output generated.")
            fb.success("PhotoGuard request completed successfully.")
            report_progress("completed", 100.0, "PhotoGuard request completed successfully.", status="completed")

            return ProcessResponse(
                requestId=request_id,
                outputs=[
                    OutputImage(
                        label=label,
                        dataUrl=_image_to_data_url(
                            image,
                            output_format=output_format,
                            lossless_output=lossless_output,
                        ),
                    )
                    for image, label in outputs
                ],
                device=PIPELINE_DEVICE,
                modelSource=INPAINT_MODEL_SOURCE,
                processingMode="immunize" if immunize else "edit",
                statusText=fb.get_status_text(),
                immunizationProfile=effective_immunization_profile,
                workingResolution=working_resolution,
                outputFormat=output_format,
                losslessOutput=lossless_output,
            )
        except HTTPException:
            _set_request_progress(
                request_id,
                status="failed",
                stage=current_stage,
                percent=current_percent,
                message="PhotoGuard request failed.",
                status_text=fb.get_status_text(),
            )
            raise
        except RuntimeError as exc:  # pragma: no cover - runtime failure path
            if _is_out_of_memory_error(exc):
                _cleanup_runtime_memory()
                fb.error("The GPU ran out of memory while processing this request.")
                fb.info("Try working resolution 512, disabling immunization, or reducing inference steps.")
                fb.info(f"Debug detail: {type(exc).__name__}")
                _set_request_progress(
                    request_id,
                    status="failed",
                    stage=current_stage,
                    percent=current_percent,
                    message="The GPU ran out of memory while processing this request.",
                    status_text=fb.get_status_text(),
                )
                raise HTTPException(status_code=503, detail=fb.get_status_text()) from exc
            if "Unable to load the inpainting model" in str(exc):
                fb.error("The inpainting model could not be loaded for this request.")
                fb.info(f"Debug detail: {type(exc).__name__}")
                _set_request_progress(
                    request_id,
                    status="failed",
                    stage=current_stage,
                    percent=current_percent,
                    message="The inpainting model could not be loaded for this request.",
                    status_text=fb.get_status_text(),
                )
                raise HTTPException(status_code=503, detail=fb.get_status_text()) from exc
            traceback.print_exc()
            fb.error("PhotoGuard hit a runtime error while processing this request.")
            fb.info(f"Debug detail: {type(exc).__name__}")
            _set_request_progress(
                request_id,
                status="failed",
                stage=current_stage,
                percent=current_percent,
                message="PhotoGuard hit a runtime error while processing this request.",
                status_text=fb.get_status_text(),
            )
            raise HTTPException(status_code=500, detail=fb.get_status_text()) from exc
        except Exception as exc:  # pragma: no cover - runtime failure path
            traceback.print_exc()
            fb.error("PhotoGuard could not finish this request.")
            fb.info(f"Debug detail: {type(exc).__name__}")
            _set_request_progress(
                request_id,
                status="failed",
                stage=current_stage,
                percent=current_percent,
                message="PhotoGuard could not finish this request.",
                status_text=fb.get_status_text(),
            )
            raise HTTPException(status_code=500, detail=fb.get_status_text()) from exc
        finally:
            _cleanup_runtime_memory()


@app.get("/progress/{request_id}", response_model=ProcessProgressResponse)
def progress(request_id: str) -> ProcessProgressResponse:
    return _get_request_progress(request_id)


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
    request_id: str = Form(""),
    prompt: str = Form(""),
    seed: str = Form(str(DEFAULT_SEED)),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(100),
    immunize: bool = Form(False),
    immunization_profile: str = Form(DEFAULT_IMMUNIZATION_PROFILE),
    working_resolution: str = Form(DEFAULT_WORKING_RESOLUTION),
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
    lossless_output: bool = Form(DEFAULT_OUTPUT_LOSSLESS),
) -> ProcessResponse:
    normalized_request_id = _normalize_request_id(request_id)
    fb = Feedback(echo=False)
    fb.info("Starting PhotoGuard request.")
    _set_request_progress(
        normalized_request_id,
        status="running",
        stage="starting",
        percent=2.0,
        message="Request accepted.",
        status_text=fb.get_status_text(),
    )
    fb.processing("Validating uploaded source image.")
    _set_request_progress(
        normalized_request_id,
        status="running",
        stage="validating",
        percent=4.0,
        message="Validating uploaded source image.",
        status_text=fb.get_status_text(),
    )
    try:
        image_bytes = await _read_upload(image, fb, label="Source image")
        _set_request_progress(
            normalized_request_id,
            status="running",
            stage="validating",
            percent=8.0,
            message="Source image validated.",
            status_text=fb.get_status_text(),
        )
        fb.processing("Validating uploaded mask image.")
        _set_request_progress(
            normalized_request_id,
            status="running",
            stage="validating",
            percent=10.0,
            message="Validating uploaded mask image.",
            status_text=fb.get_status_text(),
        )
        mask_bytes = await _read_upload(mask, fb, label="Mask image")
        normalized_seed = _normalize_seed(seed, fb)
        normalized_profile = _normalize_immunization_profile(immunization_profile, fb)
        normalized_working_resolution = _normalize_working_resolution(working_resolution, fb)
        normalized_output_format = _normalize_output_format(output_format, fb)
        if prompt.strip():
            fb.info("Prompt received and ready for processing.")
        else:
            fb.warning("No prompt provided. The edit pass will use an empty prompt.")
        _set_request_progress(
            normalized_request_id,
            status="running",
            stage="queued",
            percent=12.0,
            message="Validation complete. Waiting for the processing lock.",
            status_text=fb.get_status_text(),
        )
    except HTTPException:
        _set_request_progress(
            normalized_request_id,
            status="failed",
            stage="validating",
            percent=12.0,
            message="Request failed during validation.",
            status_text=fb.get_status_text(),
        )
        raise

    return await run_in_threadpool(
        _process_request,
        request_id=normalized_request_id,
        image_data=image_bytes,
        mask_data=mask_bytes,
        prompt=prompt,
        seed=normalized_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        immunize=immunize,
        immunization_profile=normalized_profile,
        working_resolution=normalized_working_resolution,
        output_format=normalized_output_format,
        lossless_output=lossless_output,
        fb=fb,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
