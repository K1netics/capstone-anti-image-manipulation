from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency in some envs
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, Dinov2Model
except ImportError:  # pragma: no cover
    AutoImageProcessor = None
    CLIPVisionModelWithProjection = None
    Dinov2Model = None

try:  # pragma: no cover - optional dependency in some envs
    from facenet_pytorch import InceptionResnetV1
except ImportError:  # pragma: no cover
    InceptionResnetV1 = None


@dataclass(frozen=True)
class VisionSurrogate:
    kind: str
    model_name: str
    model: torch.nn.Module
    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_VISION_CACHE: dict[tuple[str, str], VisionSurrogate | None] = {}
_FACE_ID_CACHE: dict[tuple[str, str], torch.nn.Module | None] = {}
_VISION_STATUS: dict[tuple[str, str], dict[str, object]] = {}
_FACE_ID_STATUS: dict[tuple[str, str], dict[str, object]] = {}


def _local_files_only() -> bool:
    return os.getenv("PHOTOGUARD_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _default_vision_model_name(kind: str) -> str:
    if kind == "clip":
        return os.getenv("PHOTOGUARD_CLIP_SURROGATE", "openai/clip-vit-base-patch32")
    if kind == "dino":
        return os.getenv("PHOTOGUARD_DINO_SURROGATE", "facebook/dinov2-base")
    return ""


def get_vision_surrogate_status(kind: str, device: torch.device) -> dict[str, object]:
    kind = kind.strip().lower()
    key = (kind, str(device))
    return dict(
        _VISION_STATUS.get(
            key,
            {
                "kind": kind,
                "device": str(device),
                "attempted": False,
                "loaded": False,
                "modelName": _default_vision_model_name(kind),
                "error": None,
                "localFilesOnly": _local_files_only(),
            },
        )
    )


def get_face_id_surrogate_status(device: torch.device) -> dict[str, object]:
    key = ("facenet", str(device))
    return dict(
        _FACE_ID_STATUS.get(
            key,
            {
                "kind": "facenet",
                "device": str(device),
                "attempted": False,
                "loaded": False,
                "modelName": "vggface2",
                "error": None,
            },
        )
    )


def _resolve_size(processor) -> tuple[int, int]:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, dict):
        if "height" in crop_size and "width" in crop_size:
            return int(crop_size["height"]), int(crop_size["width"])
        if "shortest_edge" in crop_size:
            edge = int(crop_size["shortest_edge"])
            return edge, edge
    return 224, 224


def _resolve_stats(processor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
    return tuple(float(x) for x in mean[:3]), tuple(float(x) for x in std[:3])


def maybe_load_vision_surrogate(kind: str, device: torch.device) -> VisionSurrogate | None:
    kind = kind.strip().lower()
    key = (kind, str(device))
    if key in _VISION_CACHE:
        return _VISION_CACHE[key]
    if AutoImageProcessor is None:
        _VISION_STATUS[key] = {
            "kind": kind,
            "device": str(device),
            "attempted": True,
            "loaded": False,
            "modelName": _default_vision_model_name(kind),
            "error": "transformers AutoImageProcessor unavailable",
            "localFilesOnly": _local_files_only(),
        }
        _VISION_CACHE[key] = None
        return None

    model_name = ""
    model = None
    try:
        if kind == "clip":
            if CLIPVisionModelWithProjection is None:
                raise RuntimeError("transformers CLIP vision model unavailable")
            model_name = os.getenv("PHOTOGUARD_CLIP_SURROGATE", "openai/clip-vit-base-patch32")
            processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=_local_files_only())
            model = CLIPVisionModelWithProjection.from_pretrained(model_name, local_files_only=_local_files_only())
        elif kind == "dino":
            if Dinov2Model is None:
                raise RuntimeError("transformers DINOv2 model unavailable")
            model_name = os.getenv("PHOTOGUARD_DINO_SURROGATE", "facebook/dinov2-base")
            processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=_local_files_only())
            model = Dinov2Model.from_pretrained(model_name, local_files_only=_local_files_only())
        else:
            raise ValueError(f"Unsupported vision surrogate kind: {kind}")
    except Exception as exc:
        _VISION_STATUS[key] = {
            "kind": kind,
            "device": str(device),
            "attempted": True,
            "loaded": False,
            "modelName": model_name or _default_vision_model_name(kind),
            "error": f"{type(exc).__name__}: {exc}",
            "localFilesOnly": _local_files_only(),
        }
        _VISION_CACHE[key] = None
        return None

    model = model.eval().to(device=device)
    model.requires_grad_(False)
    surrogate = VisionSurrogate(
        kind=kind,
        model_name=model_name,
        model=model,
        size=_resolve_size(processor),
        mean=_resolve_stats(processor)[0],
        std=_resolve_stats(processor)[1],
    )
    _VISION_STATUS[key] = {
        "kind": kind,
        "device": str(device),
        "attempted": True,
        "loaded": True,
        "modelName": model_name,
        "error": None,
        "localFilesOnly": _local_files_only(),
    }
    _VISION_CACHE[key] = surrogate
    return surrogate


def maybe_load_face_id_surrogate(device: torch.device) -> torch.nn.Module | None:
    key = ("facenet", str(device))
    if key in _FACE_ID_CACHE:
        return _FACE_ID_CACHE[key]
    if InceptionResnetV1 is None:
        _FACE_ID_STATUS[key] = {
            "kind": "facenet",
            "device": str(device),
            "attempted": True,
            "loaded": False,
            "modelName": "vggface2",
            "error": "facenet_pytorch unavailable",
        }
        _FACE_ID_CACHE[key] = None
        return None
    try:
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device=device)
        model.requires_grad_(False)
    except Exception as exc:
        _FACE_ID_STATUS[key] = {
            "kind": "facenet",
            "device": str(device),
            "attempted": True,
            "loaded": False,
            "modelName": "vggface2",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _FACE_ID_CACHE[key] = None
        return None
    _FACE_ID_STATUS[key] = {
        "kind": "facenet",
        "device": str(device),
        "attempted": True,
        "loaded": True,
        "modelName": "vggface2",
        "error": None,
    }
    _FACE_ID_CACHE[key] = model
    return model


def _normalize_mask(spatial_mask: torch.Tensor | None, size: tuple[int, int], device: torch.device) -> torch.Tensor | None:
    if spatial_mask is None:
        return None
    mask = spatial_mask.to(device=device, dtype=torch.float32)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _prepare_pixels(
    image_tensor: torch.Tensor,
    spatial_mask: torch.Tensor | None,
    *,
    output_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    fill_value: float = 0.0,
) -> torch.Tensor:
    image = image_tensor.to(dtype=torch.float32)
    # The teacher path may operate in [-1, 1], while the student/eval scripts
    # load RGB tensors in [0, 1]. Accept either without silently skewing inputs.
    if float(image.detach().amin().item()) < -0.05:
        image = ((image.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        image = image.clamp(0.0, 1.0)
    pixels = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
    if spatial_mask is not None:
        mask = _normalize_mask(spatial_mask, output_size, image.device)
        if mask is not None:
            pixels = pixels * mask + fill_value * (1.0 - mask)
    mean_t = torch.tensor(mean, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean_t) / std_t.clamp_min(1e-6)


def compute_vision_surrogate_embedding(
    surrogate: VisionSurrogate | None,
    image_tensor: torch.Tensor,
    spatial_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if surrogate is None:
        return None
    pixel_values = _prepare_pixels(
        image_tensor,
        spatial_mask,
        output_size=surrogate.size,
        mean=surrogate.mean,
        std=surrogate.std,
        fill_value=0.0,
    )
    outputs = surrogate.model(pixel_values=pixel_values)
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        embedding = outputs.image_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embedding = outputs.pooler_output
    else:
        embedding = outputs.last_hidden_state[:, 0]
    return F.normalize(embedding.to(dtype=torch.float32), dim=1)


def compute_face_id_embedding(
    model: torch.nn.Module | None,
    image_tensor: torch.Tensor,
    spatial_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if model is None or spatial_mask is None:
        return None
    pixels = _prepare_pixels(
        image_tensor,
        spatial_mask,
        output_size=(160, 160),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_value=0.0,
    )
    embedding = model(pixels)
    return F.normalize(embedding.to(dtype=torch.float32), dim=1)
