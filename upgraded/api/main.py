from __future__ import annotations

import base64
import gc
import inspect
import io
import json
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import torch
from diffusers import AutoPipelineForInpainting
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

CURRENT_FILE = Path(__file__).resolve()
UPGRADED_ROOT = CURRENT_FILE.parents[1]
LOCAL_BACKEND_ROOT = UPGRADED_ROOT / "backend"
RUNPOD_WORKSPACE_MODEL_DIR = Path("/workspace").resolve()


def _configured_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _has_local_model_files(model_dir: Path) -> bool:
    return model_dir.is_dir() and (model_dir / "model_index.json").exists()


def _candidate_local_model_dirs() -> list[Path]:
    configured_model_dir = _configured_path("PHOTOGUARD_MODEL_DIR")
    candidates: list[Path] = []
    if configured_model_dir is not None:
        candidates.append(configured_model_dir)

    candidates.extend([
        Path("/workspace"),
        Path("/workspace/sdxl_local_inpaint_model"),
        Path("/workspace/local_inpaint_model"),
        Path("/workspace/models/local_inpaint_model"),
        Path("/workspace/artifacts/local_inpaint_model"),
        Path("/workspace/photoguard/artifacts/local_inpaint_model"),
        UPGRADED_ROOT / "artifacts" / "local_inpaint_model",
    ])

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.expanduser().resolve())
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(Path(normalized))
    return deduped


def _default_local_model_dir() -> Path:
    configured_model_dir = _configured_path("PHOTOGUARD_MODEL_DIR")
    if configured_model_dir is not None:
        if _has_local_model_files(configured_model_dir):
            return configured_model_dir
        return configured_model_dir

    if RUNPOD_WORKSPACE_MODEL_DIR.exists():
        return RUNPOD_WORKSPACE_MODEL_DIR

    for candidate in _candidate_local_model_dirs():
        if _has_local_model_files(candidate):
            return candidate

    return UPGRADED_ROOT / "artifacts" / "local_inpaint_model"


if str(LOCAL_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_BACKEND_ROOT))

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
SUPPORTED_DEFENSE_CANVASES = ("profile_default", "512", "640", "768", "1024", "working")
DELEGATED_IMMUNIZATION_PROFILES: dict[str, str] = {}
REQUIRE_FULL_STRENGTH_IMMUNIZATION = os.getenv(
    "PHOTOGUARD_REQUIRE_FULL_STRENGTH",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}

DEFAULT_IMMUNIZATION_CONFIG = ImmunizationConfig(
    profile_name="stable_diffusion",
    eps=0.085,
    step_size=0.0065,
    target_mode="random",
    iters=28,
    target_strength=0.9,
    chaos_strength=0.28,
    denoiser_strength=0.16,
    denoiser_steps=2,
    denoiser_early_timestep_bias=1.4,
    eot_samples=2,
    resize_jitter=0.08,
    noise_strength=0.015,
    blur_kernel_size=3,
    mask_augmentation_strength=0.16,
    mask_augmentation_count=1,
    semantic_boundary_strength=0.08,
    semantic_ring_width=8,
    region_priority_strength=0.05,
    priority_face_strength=0.35,
    priority_skin_strength=0.22,
    priority_text_strength=0.16,
    priority_logo_strength=0.16,
    watermark_strength=0.035,
    reference_confusion_strength=0.06,
    identity_drift_strength=0.05,
    portrait_face_identity_strength=0.05,
    compression_jitter_strength=0.03,
    subpixel_jitter=0.2,
    frequency_noise_strength=0.0025,
    tripwire_global_strength=0.04,
    tripwire_global_count=2,
    max_prompt_variants=6,
    low_memory_loss_scale=0.45,
    allow_low_memory_fallback=not REQUIRE_FULL_STRENGTH_IMMUNIZATION,
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
NANO_BANANA_2_IMMUNIZATION_RESOLUTION = 384


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


PIPELINE_COMPILE_ENABLED = _env_flag("PHOTOGUARD_COMPILE", False)
PIPELINE_XFORMERS_ENABLED = _env_flag("PHOTOGUARD_ENABLE_XFORMERS", True)


DEFAULT_MODEL_SOURCE = (
    str(DEFAULT_LOCAL_MODEL_DIR)
    if DEFAULT_LOCAL_MODEL_DIR == RUNPOD_WORKSPACE_MODEL_DIR or _has_local_model_files(DEFAULT_LOCAL_MODEL_DIR)
    else DEFAULT_INPAINT_MODEL_ID
)
INPAINT_MODEL_SOURCE = os.getenv("PHOTOGUARD_INPAINT_MODEL", DEFAULT_MODEL_SOURCE)
INPAINT_LOCAL_FILES_ONLY = _env_flag(
    "PHOTOGUARD_LOCAL_FILES_ONLY",
    DEFAULT_LOCAL_MODEL_DIR == RUNPOD_WORKSPACE_MODEL_DIR or _has_local_model_files(DEFAULT_LOCAL_MODEL_DIR),
)
DEFAULT_OUTPUT_LOSSLESS = _env_flag("PHOTOGUARD_OUTPUT_LOSSLESS", True)
PIPELINE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DTYPE = torch.float16 if PIPELINE_DEVICE == "cuda" else torch.float32

_pipeline: object | None = None
_pipeline_error: str | None = None
_pipeline_runtime_features: dict[str, object] = {}
_pipeline_init_lock = Lock()
_pipeline_run_lock = Lock()
_request_progress_lock = Lock()
_request_progress: dict[str, dict[str, object]] = {}
_request_results_lock = Lock()
_request_results: dict[str, "ProcessResponse"] = {}

METRIC_LABELS = {
    "anchors": "Sparse anchors",
    "bg_anchor": "Context anchor confusion",
    "chaos": "Latent drift",
    "context": "Boundary context blend",
    "denoiser": "Proxy denoiser drift",
    "global": "Global anchor",
    "identity": "Identity drift",
    "loss": "Total loss",
    "ms_desc": "Multi-scale descriptor",
    "peakAllocGiB": "Peak VRAM alloc GiB",
    "peakReservedGiB": "Peak VRAM reserved GiB",
    "priority": "Priority regions",
    "protected": "Protected region change",
    "reference": "Reference confusion",
    "ring": "Boundary ring",
    "sd_bnd_confusion": "SD boundary confusion",
    "sd_confusion": "SD subject confusion",
    "sd_drift": "SD identity drift",
    "sd_face": "SD face identity confusion",
    "sd_face_drift": "SD face drift",
    "sd_global": "SD context anchors",
    "semantic": "Semantic boundary",
    "stepSize": "Step size",
    "target": "Target latent",
    "tripwire": "Protected tripwire",
    "watermark": "Frequency watermark",
}


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


class ProcessStartResponse(BaseModel):
    requestId: str
    status: str
    statusText: str


class HealthResponse(BaseModel):
    status: str
    device: str
    modelSource: str
    pipelineReady: bool
    pipelineError: str | None = None


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


@dataclass(frozen=True)
class BackendSettingOverrides:
    defense_canvas: str = "profile_default"
    force_full_strength: bool = True
    immunization_iters: int | None = None
    eot_samples: int | None = None
    max_prompt_variants: int | None = None
    denoiser_strength: float | None = None
    reference_confusion_strength: float | None = None
    identity_drift_strength: float | None = None
    semantic_boundary_strength: float | None = None
    watermark_strength: float | None = None
    tripwire_global_strength: float | None = None


@dataclass(frozen=True)
class PreparedProcessRequest:
    request_id: str
    image_data: bytes
    mask_data: bytes
    prompt: str
    seed: int
    guidance_scale: float
    num_inference_steps: int
    immunize: bool
    immunization_profile: str
    working_resolution: str
    output_format: str
    lossless_output: bool
    backend_overrides: BackendSettingOverrides


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


def _gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


def _get_cuda_memory_metrics() -> dict[str, float]:
    if PIPELINE_DEVICE != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "allocatedGiB": _gib(torch.cuda.memory_allocated()),
        "reservedGiB": _gib(torch.cuda.memory_reserved()),
        "peakAllocGiB": _gib(torch.cuda.max_memory_allocated()),
        "peakReservedGiB": _gib(torch.cuda.max_memory_reserved()),
    }


def _reset_cuda_peak_memory_stats() -> None:
    if PIPELINE_DEVICE != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _append_cuda_memory_feedback(fb: Feedback, prefix: str) -> None:
    if PIPELINE_DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    stats = _get_cuda_memory_metrics()
    if not stats:
        return
    fb.info(
        f"{prefix} peak VRAM: allocated {stats['peakAllocGiB']:.2f} GiB, "
        f"reserved {stats['peakReservedGiB']:.2f} GiB."
    )


def _report_pipeline_runtime_features(fb: Feedback) -> None:
    family = str(_pipeline_runtime_features.get("family", "")).strip()
    if family:
        fb.info(f"Inpainting pipeline family: {family}.")
    load_variant = str(_pipeline_runtime_features.get("load_variant", "")).strip()
    if load_variant:
        fb.info(f"Inpainting weights variant: {load_variant}.")
    if PIPELINE_DEVICE != "cuda":
        return
    if _pipeline_runtime_features.get("xformers_enabled"):
        fb.info("xFormers memory-efficient attention is enabled.")
    else:
        xformers_error = str(_pipeline_runtime_features.get("xformers_error", "")).strip()
        if xformers_error:
            fb.warning(
                f"xFormers memory-efficient attention is unavailable ({xformers_error}); using attention slicing instead."
            )
        else:
            fb.warning("xFormers memory-efficient attention is unavailable; using attention slicing instead.")
    if _pipeline_runtime_features.get("unet_gradient_checkpointing"):
        fb.info("UNet gradient checkpointing is enabled for lower immunization VRAM.")
    if _pipeline_runtime_features.get("vae_gradient_checkpointing"):
        fb.info("VAE gradient checkpointing is enabled for lower immunization VRAM.")
    if _pipeline_runtime_features.get("compile_enabled"):
        fb.info("torch.compile is enabled for the UNet.")


def _is_sdxl_pipeline(pipeline: object) -> bool:
    class_name = pipeline.__class__.__name__.lower()
    if "xl" in class_name:
        return True
    return getattr(pipeline, "text_encoder_2", None) is not None


def _describe_pipeline_family(pipeline: object) -> str:
    return "sdxl_inpaint" if _is_sdxl_pipeline(pipeline) else "stable_diffusion_inpaint"


def _model_source_path(model_source: str) -> Path | None:
    candidate = Path(model_source).expanduser()
    if not candidate.exists():
        return None
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _read_model_index(model_dir: Path) -> dict[str, object] | None:
    model_index_path = model_dir / "model_index.json"
    if not model_index_path.exists():
        return None
    try:
        return json.loads(model_index_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_sdxl_model_index(model_index: dict[str, object] | None) -> bool:
    if not isinstance(model_index, dict):
        return False
    class_name = str(model_index.get("_class_name", "")).lower()
    return "xl" in class_name or "sdxl" in class_name


def _local_model_validation_error(model_dir: Path) -> str | None:
    model_index_path = model_dir / "model_index.json"
    if not model_index_path.exists():
        nested_model = next((candidate.parent for candidate in model_dir.glob("*/model_index.json")), None)
        if nested_model is not None:
            return (
                f"Local model root {str(model_dir)!r} does not contain model_index.json. "
                f"Found a nested model directory at {str(nested_model)!r}; point PHOTOGUARD_INPAINT_MODEL there "
                "or move the snapshot contents directly into the root path."
            )
        return (
            f"Local model root {str(model_dir)!r} does not contain model_index.json yet. "
            "Upload a full Diffusers snapshot into that path before sending a request."
        )

    model_index = _read_model_index(model_dir)
    if _is_sdxl_model_index(model_index):
        required_entries = (
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "scheduler",
        )
        missing_entries = [entry for entry in required_entries if not (model_dir / entry).exists()]
        if missing_entries:
            missing_display = ", ".join(missing_entries)
            return (
                f"Local SDXL model root {str(model_dir)!r} is missing required entries: {missing_display}. "
                "The official SDXL inpainting snapshot needs both text encoders and both tokenizers."
            )
    return None


def _local_model_uses_fp16_variant(model_dir: Path) -> bool:
    if not _is_sdxl_model_index(_read_model_index(model_dir)):
        return False
    for pattern in ("*.fp16.safetensors", "*.fp16.bin"):
        if next(model_dir.rglob(pattern), None) is not None:
            return True
    return False


def _load_pipeline_from_source(model_source: str) -> tuple[object, dict[str, object]]:
    source_path = _model_source_path(model_source)
    if source_path is not None and source_path.is_dir():
        validation_error = _local_model_validation_error(source_path)
        if validation_error is not None:
            raise RuntimeError(validation_error)

    load_attempts: list[dict[str, object]] = []
    if source_path is not None and source_path.is_dir() and _local_model_uses_fp16_variant(source_path):
        load_attempts.append({
            "variant": "fp16",
            "use_safetensors": True,
            "hint": "fp16",
        })
    load_attempts.append({"hint": "default"})

    errors: list[str] = []
    for attempt in load_attempts:
        load_kwargs: dict[str, object] = {
            "local_files_only": INPAINT_LOCAL_FILES_ONLY,
            "torch_dtype": PIPELINE_DTYPE,
        }
        if attempt.get("variant") is not None:
            load_kwargs["variant"] = attempt["variant"]
        if attempt.get("use_safetensors") is not None:
            load_kwargs["use_safetensors"] = attempt["use_safetensors"]
        try:
            pipeline = AutoPipelineForInpainting.from_pretrained(
                model_source,
                **load_kwargs,
            )
            return pipeline, {"load_variant": str(attempt["hint"])}
        except Exception as exc:
            detail = str(exc).strip()
            errors.append(f"{attempt['hint']}: {type(exc).__name__}: {detail}")

    combined_errors = " | ".join(error[:320] for error in errors if error)
    raise RuntimeError(
        "Unable to load the inpainting model. "
        f"Current source: {model_source!r}. "
        f"Attempt details: {combined_errors}"
    )


def _build_pipeline_call_kwargs(
    pipeline: object,
    *,
    prompt: str,
    image: Image.Image,
    mask_image: Image.Image,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> dict[str, object]:
    call_params = inspect.signature(pipeline.__call__).parameters
    call_kwargs: dict[str, object] = {
        "prompt": prompt,
        "image": image,
        "mask_image": mask_image,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    if "eta" in call_params:
        call_kwargs["eta"] = 1
    if _is_sdxl_pipeline(pipeline) and "strength" in call_params:
        # SDXL inpainting behaves more reliably when not forced to the exact
        # strength=1 boundary used by older SD2-style inpaint setups.
        call_kwargs["strength"] = 0.99
    return call_kwargs


def _configure_pipeline_memory(pipeline: object) -> dict[str, object]:
    features: dict[str, object] = {
        "xformers_enabled": False,
        "attention_slicing_enabled": False,
        "unet_gradient_checkpointing": False,
        "vae_gradient_checkpointing": False,
        "compile_enabled": False,
        "family": _describe_pipeline_family(pipeline),
    }
    if PIPELINE_DEVICE == "cuda":
        if PIPELINE_XFORMERS_ENABLED:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                features["xformers_enabled"] = True
            except Exception as exc:
                features["xformers_error"] = type(exc).__name__
        try:
            pipeline.vae.enable_slicing()
            features["vae_slicing_enabled"] = True
        except Exception:
            pass
        try:
            pipeline.vae.enable_tiling()
            features["vae_tiling_enabled"] = True
        except Exception:
            pass
        try:
            if hasattr(pipeline.unet, "enable_gradient_checkpointing"):
                pipeline.unet.enable_gradient_checkpointing()
                features["unet_gradient_checkpointing"] = True
        except Exception as exc:
            features["unet_gradient_checkpointing_error"] = type(exc).__name__
        try:
            if hasattr(pipeline.vae, "enable_gradient_checkpointing"):
                pipeline.vae.enable_gradient_checkpointing()
                features["vae_gradient_checkpointing"] = True
        except Exception as exc:
            features["vae_gradient_checkpointing_error"] = type(exc).__name__

    if PIPELINE_DEVICE == "cpu" or not features["xformers_enabled"]:
        pipeline.enable_attention_slicing()
        features["attention_slicing_enabled"] = True

    if PIPELINE_COMPILE_ENABLED and hasattr(torch, "compile"):
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            features["compile_enabled"] = True
        except Exception:
            pass
    return features


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _is_retryable_gpu_runtime_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "cuda",
            "cublas_status_alloc_failed",
            "cublas_status_not_initialized",
            "cublas",
            "cudnn_status_alloc_failed",
            "cudnn",
            "cuda error: an illegal memory access was encountered",
            "cuda error: device-side assert triggered",
            "cuda error: unspecified launch failure",
        )
    )


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


def _set_request_result(request_id: str, result: ProcessResponse) -> None:
    with _request_results_lock:
        _request_results[request_id] = result


def _get_request_result(request_id: str) -> ProcessResponse | None:
    with _request_results_lock:
        return _request_results.get(request_id)


def _clear_request_state(request_id: str) -> None:
    with _request_progress_lock:
        _request_progress.pop(request_id, None)
    with _request_results_lock:
        _request_results.pop(request_id, None)


def _get_pipeline() -> object:
    global _pipeline, _pipeline_error, _pipeline_runtime_features

    if _pipeline is not None:
        return _pipeline

    with _pipeline_init_lock:
        if _pipeline is not None:
            return _pipeline

        try:
            pipeline, load_metadata = _load_pipeline_from_source(INPAINT_MODEL_SOURCE)
            if hasattr(pipeline, "safety_checker"):
                try:
                    pipeline.safety_checker = None
                except Exception:
                    pass
            pipeline = pipeline.to(PIPELINE_DEVICE)
            _pipeline_runtime_features = _configure_pipeline_memory(pipeline)
            _pipeline_runtime_features.update(load_metadata)
            _pipeline = pipeline
            _pipeline_error = None
        except Exception as exc:  # pragma: no cover - startup failure path
            _pipeline_error = str(exc)
            raise RuntimeError(str(exc)) from exc

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


def _normalize_defense_canvas(defense_canvas: str | None, fb: Feedback) -> str:
    normalized = (defense_canvas or "profile_default").strip().lower()
    if normalized not in SUPPORTED_DEFENSE_CANVASES:
        fb.error("Defense canvas must be one of profile_default, 512, 640, 768, 1024, or working.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    if normalized == "working":
        fb.info("Backend override: using the full working canvas for immunization.")
    elif normalized != "profile_default":
        fb.info(f"Backend override: using a {normalized}px defense canvas.")
    return normalized


def _parse_optional_int_override(
    value: str | None,
    *,
    field_label: str,
    minimum: int,
    maximum: int | None,
    fb: Feedback,
) -> int | None:
    stripped = (value or "").strip()
    if not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError as exc:
        fb.error(f"{field_label} must be an integer.")
        raise HTTPException(status_code=400, detail=fb.get_status_text()) from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            fb.error(f"{field_label} must be at least {minimum}.")
        else:
            fb.error(f"{field_label} must be between {minimum} and {maximum}.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    fb.info(f"Backend override: {field_label} set to {parsed}.")
    return parsed


def _parse_optional_float_override(
    value: str | None,
    *,
    field_label: str,
    minimum: float,
    maximum: float | None,
    fb: Feedback,
) -> float | None:
    stripped = (value or "").strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError as exc:
        fb.error(f"{field_label} must be a number.")
        raise HTTPException(status_code=400, detail=fb.get_status_text()) from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            fb.error(f"{field_label} must be at least {minimum}.")
        else:
            fb.error(f"{field_label} must be between {minimum} and {maximum}.")
        raise HTTPException(status_code=400, detail=fb.get_status_text())
    fb.info(f"Backend override: {field_label} set to {parsed:.3f}.")
    return parsed


def _normalize_backend_setting_overrides(
    *,
    defense_canvas: str,
    force_full_strength: bool,
    immunization_iters: str,
    eot_samples: str,
    max_prompt_variants: str,
    denoiser_strength: str,
    reference_confusion_strength: str,
    identity_drift_strength: str,
    semantic_boundary_strength: str,
    watermark_strength: str,
    tripwire_global_strength: str,
    fb: Feedback,
) -> BackendSettingOverrides:
    normalized_canvas = _normalize_defense_canvas(defense_canvas, fb)
    if force_full_strength:
        fb.info("Backend override: full-strength fail-closed mode is enabled.")
    else:
        fb.warning("Backend override: low-memory fallback is allowed for this run.")
    return BackendSettingOverrides(
        defense_canvas=normalized_canvas,
        force_full_strength=force_full_strength,
        immunization_iters=_parse_optional_int_override(
            immunization_iters,
            field_label="PGD iterations",
            minimum=1,
            maximum=500,
            fb=fb,
        ),
        eot_samples=_parse_optional_int_override(
            eot_samples,
            field_label="EOT samples",
            minimum=1,
            maximum=16,
            fb=fb,
        ),
        max_prompt_variants=_parse_optional_int_override(
            max_prompt_variants,
            field_label="Prompt variants",
            minimum=1,
            maximum=16,
            fb=fb,
        ),
        denoiser_strength=_parse_optional_float_override(
            denoiser_strength,
            field_label="Denoiser strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
        reference_confusion_strength=_parse_optional_float_override(
            reference_confusion_strength,
            field_label="Reference confusion strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
        identity_drift_strength=_parse_optional_float_override(
            identity_drift_strength,
            field_label="Identity drift strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
        semantic_boundary_strength=_parse_optional_float_override(
            semantic_boundary_strength,
            field_label="Semantic boundary strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
        watermark_strength=_parse_optional_float_override(
            watermark_strength,
            field_label="Watermark strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
        tripwire_global_strength=_parse_optional_float_override(
            tripwire_global_strength,
            field_label="Global anchor strength",
            minimum=0.0,
            maximum=1.0,
            fb=fb,
        ),
    )


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


def _preferred_immunization_resolution(profile_name: str) -> int:
    if profile_name in {"nano_banana_2", "nano_banana_2_hard_block", "nano_banana_2_distortion"}:
        return NANO_BANANA_2_IMMUNIZATION_RESOLUTION
    return DEFAULT_IMMUNIZATION_RESOLUTION


def _resolve_immunization_resolution(
    profile_name: str,
    working_image_size: tuple[int, int],
    defense_canvas: str,
) -> int:
    if defense_canvas == "profile_default":
        return _preferred_immunization_resolution(profile_name)
    if defense_canvas == "working":
        return max(working_image_size)
    return int(defense_canvas)


def _is_strict_immunization_profile(profile_name: str) -> bool:
    return profile_name in {"nano_banana_2_hard_block", "nano_banana_2_distortion"}


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
    if profile_name == "full_regeneration_scaffold":
        return replace(
            config,
            target_mode="shifted_input",
            iters=12,
            target_strength=0.55,
            chaos_strength=0.18,
            denoiser_strength=0.12,
            resize_jitter=0.05,
            noise_strength=0.004,
            compression_jitter_strength=0.03,
            subpixel_jitter=0.15,
            frequency_noise_strength=0.002,
            watermark_strength=0.03,
            region_priority_strength=0.03,
            priority_face_strength=0.15,
            priority_skin_strength=0.1,
            semantic_boundary_strength=0.02,
            max_prompt_variants=3,
        )
    if profile_name == "instruction_editing_scaffold":
        return replace(
            config,
            target_mode="shifted_input",
            iters=12,
            target_strength=0.5,
            chaos_strength=0.16,
            denoiser_strength=0.12,
            resize_jitter=0.04,
            noise_strength=0.004,
            compression_jitter_strength=0.035,
            subpixel_jitter=0.2,
            frequency_noise_strength=0.002,
            watermark_strength=0.04,
            region_priority_strength=0.05,
            priority_face_strength=0.2,
            priority_skin_strength=0.15,
            semantic_boundary_strength=0.04,
            max_prompt_variants=3,
        )
    if profile_name == "controlnet_scaffold":
        return replace(
            config,
            target_mode="gray",
            iters=11,
            target_strength=0.42,
            chaos_strength=0.14,
            denoiser_strength=0.08,
            resize_jitter=0.03,
            noise_strength=0.003,
            compression_jitter_strength=0.03,
            subpixel_jitter=0.18,
            frequency_noise_strength=0.003,
            frequency_band_low=0.14,
            frequency_band_high=0.32,
            watermark_strength=0.03,
            region_priority_strength=0.04,
            priority_face_strength=0.18,
            priority_skin_strength=0.12,
            texture_tracking_strength=0.05,
            semantic_boundary_strength=0.03,
            max_prompt_variants=3,
        )
    if profile_name == "style_transfer_scaffold":
        return replace(
            config,
            target_mode="shifted_input",
            iters=10,
            target_strength=0.4,
            chaos_strength=0.12,
            denoiser_strength=0.08,
            resize_jitter=0.03,
            noise_strength=0.003,
            compression_jitter_strength=0.025,
            subpixel_jitter=0.12,
            frequency_noise_strength=0.0025,
            frequency_band_low=0.1,
            frequency_band_high=0.22,
            watermark_strength=0.03,
            edge_tracking_strength=0.05,
            max_prompt_variants=3,
        )
    if profile_name == "text_aware_scaffold":
        return replace(
            config,
            target_mode="gray",
            iters=11,
            target_strength=0.44,
            chaos_strength=0.13,
            denoiser_strength=0.08,
            resize_jitter=0.02,
            noise_strength=0.003,
            compression_jitter_strength=0.035,
            subpixel_jitter=0.15,
            frequency_noise_strength=0.003,
            frequency_band_low=0.12,
            frequency_band_high=0.3,
            watermark_strength=0.05,
            region_priority_strength=0.06,
            priority_text_strength=0.85,
            priority_logo_strength=0.7,
            semantic_boundary_strength=0.03,
            max_prompt_variants=3,
        )
    if profile_name == "adversarial_hardened_scaffold":
        return replace(
            config,
            target_mode="checkerboard",
            iters=14,
            target_strength=0.6,
            chaos_strength=0.2,
            denoiser_strength=0.14,
            denoiser_steps=1,
            denoiser_early_timestep_bias=0.8,
            eot_samples=2,
            resize_jitter=0.06,
            noise_strength=0.005,
            mask_augmentation_strength=0.1,
            mask_augmentation_count=1,
            compression_jitter_strength=0.045,
            subpixel_jitter=0.22,
            frequency_noise_strength=0.0035,
            frequency_band_low=0.1,
            frequency_band_high=0.3,
            watermark_strength=0.06,
            region_priority_strength=0.06,
            priority_face_strength=0.2,
            priority_skin_strength=0.15,
            priority_text_strength=0.3,
            priority_logo_strength=0.3,
            semantic_boundary_strength=0.04,
            portrait_face_identity_strength=0.04,
            max_prompt_variants=4,
        )
    if profile_name == "nano_banana_experimental":
        return replace(
            config,
            eps=0.055,
            step_size=0.0045,
            target_mode="shifted_input",
            iters=12,
            target_strength=0.42,
            chaos_strength=0.12,
            denoiser_strength=0.08,
            denoiser_steps=1,
            denoiser_early_timestep_bias=0.6,
            eot_samples=1,
            resize_jitter=0.02,
            noise_strength=0.003,
            blur_kernel_size=3,
            mask_augmentation_strength=0.08,
            mask_augmentation_count=1,
            semantic_boundary_strength=0.05,
            semantic_ring_width=10,
            compression_jitter_strength=0.04,
            subpixel_jitter=0.25,
            frequency_noise_strength=0.003,
            frequency_band_low=0.09,
            frequency_band_high=0.24,
            watermark_strength=0.07,
            region_priority_strength=0.08,
            priority_face_strength=0.4,
            priority_skin_strength=0.25,
            priority_text_strength=0.45,
            priority_logo_strength=0.45,
            portrait_face_identity_strength=0.06,
            max_prompt_variants=5,
        )
    if profile_name == "nano_banana_2":
        return replace(
            config,
            eps=0.07,
            step_size=0.0052,
            target_mode="shifted_input",
            iters=24,
            target_strength=0.48,
            chaos_strength=0.15,
            denoiser_strength=0.0,
            denoiser_steps=0,
            eot_samples=2,
            resize_jitter=0.03,
            noise_strength=0.0035,
            blur_kernel_size=3,
            semantic_boundary_strength=0.08,
            semantic_ring_width=12,
            compression_jitter_strength=0.05,
            subpixel_jitter=0.3,
            frequency_noise_strength=0.0035,
            frequency_band_low=0.08,
            frequency_band_high=0.26,
            watermark_strength=0.09,
            region_priority_strength=0.1,
            priority_face_strength=0.6,
            priority_skin_strength=0.36,
            priority_text_strength=0.58,
            priority_logo_strength=0.58,
            reference_confusion_strength=0.15,
            identity_drift_strength=0.12,
            portrait_face_identity_strength=0.08,
            context_blend_strength=0.06,
            reference_region_count=3,
            multiscale_descriptor_strength=0.06,
            multiscale_descriptor_scales=(1.0, 0.5),
            background_anchor_confusion_strength=0.04,
            background_anchor_count=3,
            tripwire_strength=0.08,
            tripwire_ring_strength=0.05,
            tripwire_anchor_strength=0.03,
            tripwire_anchor_count=3,
            updown_scale_jitter=0.18,
            sharpen_proxy_strength=0.16,
            max_prompt_variants=10,
            low_memory_loss_scale=0.4,
        )
    if profile_name == "nano_banana_2_hard_block":
        return replace(
            config,
            eps=0.09,
            step_size=0.006,
            target_mode="shifted_input",
            iters=18,
            target_strength=0.5,
            chaos_strength=0.16,
            denoiser_strength=0.0,
            denoiser_steps=0,
            eot_samples=1,
            resize_jitter=0.03,
            noise_strength=0.004,
            blur_kernel_size=1,
            semantic_boundary_strength=0.085,
            semantic_ring_width=14,
            compression_jitter_strength=0.055,
            subpixel_jitter=0.34,
            frequency_noise_strength=0.004,
            frequency_band_low=0.08,
            frequency_band_high=0.28,
            watermark_strength=0.11,
            region_priority_strength=0.14,
            priority_face_strength=0.7,
            priority_skin_strength=0.4,
            priority_text_strength=0.7,
            priority_logo_strength=0.7,
            reference_confusion_strength=0.18,
            identity_drift_strength=0.16,
            portrait_face_identity_strength=0.1,
            context_blend_strength=0.08,
            reference_region_count=3,
            tripwire_strength=0.24,
            tripwire_ring_strength=0.16,
            tripwire_anchor_strength=0.14,
            tripwire_anchor_count=5,
            tripwire_tile_size=32,
            max_prompt_variants=4,
        )
    if profile_name == "nano_banana_2_distortion":
        return replace(
            config,
            eps=0.11,
            step_size=0.007,
            target_mode="shifted_input",
            iters=20,
            target_strength=0.54,
            chaos_strength=0.18,
            denoiser_strength=0.0,
            denoiser_steps=0,
            eot_samples=1,
            resize_jitter=0.035,
            noise_strength=0.0045,
            blur_kernel_size=1,
            semantic_boundary_strength=0.095,
            semantic_ring_width=15,
            compression_jitter_strength=0.06,
            subpixel_jitter=0.36,
            frequency_noise_strength=0.0045,
            frequency_band_low=0.08,
            frequency_band_high=0.3,
            watermark_strength=0.13,
            region_priority_strength=0.16,
            priority_face_strength=0.76,
            priority_skin_strength=0.45,
            priority_text_strength=0.76,
            priority_logo_strength=0.76,
            reference_confusion_strength=0.22,
            identity_drift_strength=0.18,
            portrait_face_identity_strength=0.11,
            context_blend_strength=0.09,
            reference_region_count=3,
            tripwire_strength=0.3,
            tripwire_ring_strength=0.2,
            tripwire_anchor_strength=0.18,
            tripwire_anchor_count=6,
            tripwire_global_strength=0.16,
            tripwire_global_count=4,
            tripwire_tile_size=32,
            max_prompt_variants=4,
        )
    return config


def _low_memory_immunization_config(config: ImmunizationConfig) -> ImmunizationConfig:
    if not config.allow_low_memory_fallback:
        return config
    scale = max(0.25, float(getattr(config, "low_memory_loss_scale", 0.4)))
    return replace(
        config,
        iters=min(config.iters, 10),
        denoiser_strength=config.denoiser_strength * scale,
        denoiser_steps=min(config.denoiser_steps, 1),
        denoiser_early_timestep_bias=min(config.denoiser_early_timestep_bias, 0.75),
        eot_samples=1,
        resize_jitter=min(config.resize_jitter, 0.03),
        noise_strength=min(config.noise_strength, 0.005),
        blur_kernel_size=min(config.blur_kernel_size, 3),
        mask_augmentation_strength=min(config.mask_augmentation_strength, 0.08) * scale,
        mask_augmentation_count=min(config.mask_augmentation_count, 1),
        semantic_ring_width=min(config.semantic_ring_width, 8),
        compression_jitter_strength=min(config.compression_jitter_strength, 0.02),
        subpixel_jitter=min(config.subpixel_jitter, 0.12),
        frequency_noise_strength=min(config.frequency_noise_strength, 0.0015),
        watermark_strength=min(config.watermark_strength, 0.03) * scale,
        region_priority_strength=min(config.region_priority_strength, 0.04) * scale,
        texture_tracking_strength=min(config.texture_tracking_strength, 0.03),
        edge_tracking_strength=min(config.edge_tracking_strength, 0.03),
        reference_confusion_strength=min(config.reference_confusion_strength, 0.06) * scale,
        identity_drift_strength=min(config.identity_drift_strength, 0.05) * scale,
        portrait_face_identity_strength=min(config.portrait_face_identity_strength, 0.05) * scale,
        context_blend_strength=min(config.context_blend_strength, 0.03) * scale,
        reference_region_count=min(config.reference_region_count, 2),
        tripwire_strength=0.0 if config.profile_name in {"nano_banana_2_hard_block", "nano_banana_2_distortion"} else min(config.tripwire_strength, 0.12),
        tripwire_ring_strength=0.0 if config.profile_name in {"nano_banana_2_hard_block", "nano_banana_2_distortion"} else min(config.tripwire_ring_strength, 0.08),
        tripwire_anchor_strength=0.0 if config.profile_name in {"nano_banana_2_hard_block", "nano_banana_2_distortion"} else min(config.tripwire_anchor_strength, 0.08),
        tripwire_anchor_count=0 if config.profile_name in {"nano_banana_2_hard_block", "nano_banana_2_distortion"} else min(config.tripwire_anchor_count, 2),
        tripwire_global_strength=0.0 if config.profile_name == "nano_banana_2_distortion" else min(config.tripwire_global_strength, 0.08),
        tripwire_global_count=0 if config.profile_name == "nano_banana_2_distortion" else min(config.tripwire_global_count, 2),
        multiscale_descriptor_strength=min(config.multiscale_descriptor_strength, 0.04) * scale,
        multiscale_descriptor_scales=(1.0,),
        background_anchor_confusion_strength=min(config.background_anchor_confusion_strength, 0.03) * scale,
        background_anchor_count=min(config.background_anchor_count, 2),
        updown_scale_jitter=min(config.updown_scale_jitter, 0.12),
        sharpen_proxy_strength=min(config.sharpen_proxy_strength, 0.08),
        max_prompt_variants=min(max(1, config.max_prompt_variants), 2),
    )


def _apply_backend_setting_overrides(
    config: ImmunizationConfig,
    overrides: BackendSettingOverrides,
) -> ImmunizationConfig:
    updates: dict[str, object] = {
        "allow_low_memory_fallback": not overrides.force_full_strength,
    }
    if overrides.immunization_iters is not None:
        updates["iters"] = overrides.immunization_iters
    if overrides.eot_samples is not None:
        updates["eot_samples"] = overrides.eot_samples
    if overrides.max_prompt_variants is not None:
        updates["max_prompt_variants"] = overrides.max_prompt_variants
    if overrides.denoiser_strength is not None:
        updates["denoiser_strength"] = overrides.denoiser_strength
    if overrides.reference_confusion_strength is not None:
        updates["reference_confusion_strength"] = overrides.reference_confusion_strength
    if overrides.identity_drift_strength is not None:
        updates["identity_drift_strength"] = overrides.identity_drift_strength
    if overrides.semantic_boundary_strength is not None:
        updates["semantic_boundary_strength"] = overrides.semantic_boundary_strength
    if overrides.watermark_strength is not None:
        updates["watermark_strength"] = overrides.watermark_strength
    if overrides.tripwire_global_strength is not None:
        updates["tripwire_global_strength"] = overrides.tripwire_global_strength
    return replace(config, **updates)


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
    pipeline: object,
    *,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    config: ImmunizationConfig,
    seed: int,
    progress_callback,
    error_callback: Callable[[BaseException], None] | None = None,
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
    except Exception as exc:
        if not _is_retryable_gpu_runtime_error(exc):
            raise
        if error_callback is not None:
            error_callback(exc)
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
    backend_overrides: BackendSettingOverrides,
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
            raw_metrics = dict(update.get("metrics", {})) if isinstance(update.get("metrics"), dict) else {}
            metric_payload = {
                METRIC_LABELS.get(name, name): value
                for name, value in raw_metrics.items()
            }
            cuda_metrics = _get_cuda_memory_metrics()
            if cuda_metrics:
                metric_payload[METRIC_LABELS["peakAllocGiB"]] = cuda_metrics["peakAllocGiB"]
                metric_payload[METRIC_LABELS["peakReservedGiB"]] = cuda_metrics["peakReservedGiB"]
            if isinstance(update.get("loss"), (int, float)):
                metric_payload[METRIC_LABELS["loss"]] = float(update["loss"])
            if isinstance(update.get("stepSize"), (int, float)):
                metric_payload[METRIC_LABELS["stepSize"]] = float(update["stepSize"])
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
            _report_pipeline_runtime_features(fb)
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
                preferred_immunization_resolution = _resolve_immunization_resolution(
                    effective_immunization_profile,
                    init_image.size,
                    backend_overrides.defense_canvas,
                )
                _reset_cuda_peak_memory_stats()
                immunization_image, immunization_mask = _resize_for_immunization_fallback(
                    init_image,
                    mask_image,
                    max_dimension=preferred_immunization_resolution,
                )
                if immunization_image.size != init_image.size:
                    if effective_immunization_profile in {"nano_banana_2", "nano_banana_2_hard_block", "nano_banana_2_distortion"}:
                        fb.info(
                            "Using a separate 384px defense canvas for the Nano Banana 2 defense so the stronger profile-specific losses can run without immediately falling back."
                        )
                    else:
                        fb.info(
                            "Using a separate 512px defense canvas for immunization to reduce runtime and GPU memory use."
                        )
                immunization_config = _immunization_config_for_profile(
                    effective_immunization_profile,
                    immunization_image.size,
                )
                immunization_config = _apply_backend_setting_overrides(
                    immunization_config,
                    backend_overrides,
                )
                if not immunization_config.allow_low_memory_fallback:
                    fb.info(
                        "Full-strength evaluation mode is enabled. Immunization will fail instead of silently switching to a lower-memory defense."
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
                if immunized_working_image is None and _is_strict_immunization_profile(effective_immunization_profile):
                    _append_cuda_memory_feedback(fb, "Immunization")
                    fb.error(
                        "Nano Banana 2 hard-block could not run at full defense strength on the current GPU configuration."
                    )
                    fb.info(
                        "Hard-block mode fails closed instead of silently weakening the defense. Try a smaller source image or working resolution 512."
                    )
                    raise HTTPException(status_code=503, detail=fb.get_status_text())
                if immunized_working_image is None and not immunization_config.allow_low_memory_fallback:
                    _append_cuda_memory_feedback(fb, "Immunization")
                    fb.error(
                        "Full-strength immunization ran out of GPU memory before any lower-memory retry was allowed."
                    )
                    fb.info(
                        "This upgraded app now fails closed for evaluation. Reduce the working resolution, free GPU memory, or set PHOTOGUARD_REQUIRE_FULL_STRENGTH=0 to re-enable fallback."
                    )
                    raise HTTPException(status_code=503, detail=fb.get_status_text())
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
                    _append_cuda_memory_feedback(fb, "Immunization")
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
                _append_cuda_memory_feedback(fb, "Immunization")
                fb.success("Immunized image generated.")
                report_progress("immunizing", 82.0, "Immunized image generated.")
            else:
                effective_immunization_profile = immunization_profile

            fb.processing("Running the edit pass.")
            report_progress("editing", 88.0, "Running the edit pass.")
            _reset_cuda_peak_memory_stats()
            edited_image = pipeline(
                **_build_pipeline_call_kwargs(
                    pipeline,
                    prompt=prompt,
                    image=immunized_working_image if immunize else init_image,
                    mask_image=mask_image,
                    width=init_image.size[0],
                    height=init_image.size[1],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ),
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
            _append_cuda_memory_feedback(fb, "Edit pass")
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
                _append_cuda_memory_feedback(fb, "Request")
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
                runtime_detail = str(exc).strip()
                if runtime_detail:
                    fb.info(f"Debug detail: {runtime_detail[:320]}")
                else:
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
            runtime_detail = str(exc).strip()
            if runtime_detail:
                fb.info(f"Debug detail: {type(exc).__name__}: {runtime_detail[:220]}")
            else:
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
            runtime_detail = str(exc).strip()
            if runtime_detail:
                fb.info(f"Debug detail: {type(exc).__name__}: {runtime_detail[:220]}")
            else:
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


async def _prepare_process_request(
    *,
    image: UploadFile,
    mask: UploadFile,
    request_id: str,
    prompt: str,
    seed: str,
    guidance_scale: float,
    num_inference_steps: int,
    immunize: bool,
    immunization_profile: str,
    working_resolution: str,
    output_format: str,
    lossless_output: bool,
    defense_canvas: str,
    force_full_strength: bool,
    immunization_iters: str,
    eot_samples: str,
    max_prompt_variants: str,
    denoiser_strength: str,
    reference_confusion_strength: str,
    identity_drift_strength: str,
    semantic_boundary_strength: str,
    watermark_strength: str,
    tripwire_global_strength: str,
) -> tuple[PreparedProcessRequest, Feedback]:
    normalized_request_id = _normalize_request_id(request_id)
    _clear_request_state(normalized_request_id)
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
        normalized_backend_overrides = (
            _normalize_backend_setting_overrides(
                defense_canvas=defense_canvas,
                force_full_strength=force_full_strength,
                immunization_iters=immunization_iters,
                eot_samples=eot_samples,
                max_prompt_variants=max_prompt_variants,
                denoiser_strength=denoiser_strength,
                reference_confusion_strength=reference_confusion_strength,
                identity_drift_strength=identity_drift_strength,
                semantic_boundary_strength=semantic_boundary_strength,
                watermark_strength=watermark_strength,
                tripwire_global_strength=tripwire_global_strength,
                fb=fb,
            )
            if immunize
            else BackendSettingOverrides(force_full_strength=force_full_strength)
        )
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

    return (
        PreparedProcessRequest(
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
            backend_overrides=normalized_backend_overrides,
        ),
        fb,
    )


def _run_background_request(prepared_request: PreparedProcessRequest, fb: Feedback) -> None:
    try:
        result = _process_request(
            request_id=prepared_request.request_id,
            image_data=prepared_request.image_data,
            mask_data=prepared_request.mask_data,
            prompt=prepared_request.prompt,
            seed=prepared_request.seed,
            guidance_scale=prepared_request.guidance_scale,
            num_inference_steps=prepared_request.num_inference_steps,
            immunize=prepared_request.immunize,
            immunization_profile=prepared_request.immunization_profile,
            working_resolution=prepared_request.working_resolution,
            output_format=prepared_request.output_format,
            lossless_output=prepared_request.lossless_output,
            backend_overrides=prepared_request.backend_overrides,
            fb=fb,
        )
        _set_request_result(prepared_request.request_id, result)
    except HTTPException:
        return
    except Exception:
        return


@app.get("/progress/{request_id}", response_model=ProcessProgressResponse)
def progress(request_id: str) -> ProcessProgressResponse:
    return _get_request_progress(request_id)


@app.get("/result/{request_id}", response_model=ProcessResponse)
def result(request_id: str) -> ProcessResponse:
    stored_result = _get_request_result(request_id)
    if stored_result is not None:
        return stored_result

    current_progress = _get_request_progress(request_id)
    if current_progress.status == "failed":
        raise HTTPException(status_code=409, detail=current_progress.statusText)
    raise HTTPException(status_code=404, detail="Result not ready yet.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _pipeline_error is None else "offline",
        device=PIPELINE_DEVICE,
        modelSource=INPAINT_MODEL_SOURCE,
        pipelineReady=_pipeline is not None,
        pipelineError=_pipeline_error,
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
    defense_canvas: str = Form("profile_default"),
    force_full_strength: bool = Form(True),
    immunization_iters: str = Form(""),
    eot_samples: str = Form(""),
    max_prompt_variants: str = Form(""),
    denoiser_strength: str = Form(""),
    reference_confusion_strength: str = Form(""),
    identity_drift_strength: str = Form(""),
    semantic_boundary_strength: str = Form(""),
    watermark_strength: str = Form(""),
    tripwire_global_strength: str = Form(""),
) -> ProcessResponse:
    prepared_request, fb = await _prepare_process_request(
        image=image,
        mask=mask,
        request_id=request_id,
        prompt=prompt,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        immunize=immunize,
        immunization_profile=immunization_profile,
        working_resolution=working_resolution,
        output_format=output_format,
        lossless_output=lossless_output,
        defense_canvas=defense_canvas,
        force_full_strength=force_full_strength,
        immunization_iters=immunization_iters,
        eot_samples=eot_samples,
        max_prompt_variants=max_prompt_variants,
        denoiser_strength=denoiser_strength,
        reference_confusion_strength=reference_confusion_strength,
        identity_drift_strength=identity_drift_strength,
        semantic_boundary_strength=semantic_boundary_strength,
        watermark_strength=watermark_strength,
        tripwire_global_strength=tripwire_global_strength,
    )

    return await run_in_threadpool(
        _process_request,
        request_id=prepared_request.request_id,
        image_data=prepared_request.image_data,
        mask_data=prepared_request.mask_data,
        prompt=prepared_request.prompt,
        seed=prepared_request.seed,
        guidance_scale=prepared_request.guidance_scale,
        num_inference_steps=prepared_request.num_inference_steps,
        immunize=prepared_request.immunize,
        immunization_profile=prepared_request.immunization_profile,
        working_resolution=prepared_request.working_resolution,
        output_format=prepared_request.output_format,
        lossless_output=prepared_request.lossless_output,
        backend_overrides=prepared_request.backend_overrides,
        fb=fb,
    )


@app.post("/process/start", response_model=ProcessStartResponse)
async def process_start(
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
    defense_canvas: str = Form("profile_default"),
    force_full_strength: bool = Form(True),
    immunization_iters: str = Form(""),
    eot_samples: str = Form(""),
    max_prompt_variants: str = Form(""),
    denoiser_strength: str = Form(""),
    reference_confusion_strength: str = Form(""),
    identity_drift_strength: str = Form(""),
    semantic_boundary_strength: str = Form(""),
    watermark_strength: str = Form(""),
    tripwire_global_strength: str = Form(""),
) -> ProcessStartResponse:
    prepared_request, fb = await _prepare_process_request(
        image=image,
        mask=mask,
        request_id=request_id,
        prompt=prompt,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        immunize=immunize,
        immunization_profile=immunization_profile,
        working_resolution=working_resolution,
        output_format=output_format,
        lossless_output=lossless_output,
        defense_canvas=defense_canvas,
        force_full_strength=force_full_strength,
        immunization_iters=immunization_iters,
        eot_samples=eot_samples,
        max_prompt_variants=max_prompt_variants,
        denoiser_strength=denoiser_strength,
        reference_confusion_strength=reference_confusion_strength,
        identity_drift_strength=identity_drift_strength,
        semantic_boundary_strength=semantic_boundary_strength,
        watermark_strength=watermark_strength,
        tripwire_global_strength=tripwire_global_strength,
    )

    worker = Thread(
        target=_run_background_request,
        kwargs={"prepared_request": prepared_request, "fb": fb},
        daemon=True,
    )
    worker.start()

    return ProcessStartResponse(
        requestId=prepared_request.request_id,
        status="accepted",
        statusText=fb.get_status_text(),
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
