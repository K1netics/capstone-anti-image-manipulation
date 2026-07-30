#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.immunization import (  # noqa: E402
    ImmunizationConfig,
    apply_light_compression_proxy,
    apply_sharpen_proxy,
    apply_updown_scale_proxy,
    build_identity_drift_view,
    compute_masked_descriptor,
    descriptor_similarity,
    normalize_map,
    rgb_to_luma,
    sobel_magnitude,
)

try:  # pragma: no cover - optional dependency in local envs
    import lpips  # type: ignore
except ImportError:  # pragma: no cover
    lpips = None

try:  # pragma: no cover - optional dependency in local envs
    from facenet_pytorch import InceptionResnetV1  # type: ignore
except ImportError:  # pragma: no cover
    InceptionResnetV1 = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-pass protector model from teacher-exported SDXL samples.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to teacher manifest.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--image-size", type=int, default=512, help="Square training canvas size.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--sampling-mode",
        choices=("grouped_prompts", "flat"),
        default="grouped_prompts",
        help="Grouped prompts samples one prompt per base image each epoch to encourage prompt-agnostic protection.",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split by baseId.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on manifest rows for quick experiments.")
    parser.add_argument("--max-epsilon", type=float, default=8.0 / 255.0, help="Maximum perturbation magnitude.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count for the protector U-Net.")
    parser.add_argument("--teacher-image-weight", type=float, default=1.0, help="Weight for teacher protected image matching.")
    parser.add_argument("--teacher-delta-weight", type=float, default=0.3, help="Weight for teacher delta matching.")
    parser.add_argument("--fidelity-l1-weight", type=float, default=0.12, help="Weight for source-image L1 fidelity.")
    parser.add_argument("--lpips-weight", type=float, default=0.08, help="Weight for LPIPS fidelity if available.")
    parser.add_argument("--face-identity-weight", type=float, default=0.1, help="Weight for portrait face-identity confusion.")
    parser.add_argument("--face-drift-weight", type=float, default=0.05, help="Weight for drift-view face identity confusion.")
    parser.add_argument(
        "--face-id-backend",
        choices=("auto", "none", "facenet"),
        default="auto",
        help="Real face-ID encoder backend. 'auto' uses FaceNet if facenet-pytorch is installed.",
    )
    parser.add_argument("--real-face-id-weight", type=float, default=0.12, help="Weight for real face-ID confusion loss.")
    parser.add_argument("--real-face-id-drift-weight", type=float, default=0.06, help="Weight for purified/drifted real face-ID confusion.")
    parser.add_argument("--family-balance-alpha", type=float, default=1.0, help="Inverse-frequency balancing strength for prompt families.")
    parser.add_argument("--purification-strength", type=float, default=0.18, help="Strength of cleanup/purification proxy transforms.")
    parser.add_argument("--purification-consistency-weight", type=float, default=0.16, help="Weight for matching teacher outputs after purification transforms.")
    parser.add_argument("--purification-face-weight", type=float, default=0.05, help="Weight for face identity confusion under purification transforms.")
    parser.add_argument("--selective-visibility-weight", type=float, default=0.04, help="Weight for hiding perturbations in textured/non-face regions.")
    parser.add_argument("--visual-tv-weight", type=float, default=0.01, help="Total-variation regularizer on the perturbation.")
    parser.add_argument("--delta-l2-weight", type=float, default=0.02, help="L2 energy regularizer on the perturbation.")
    parser.add_argument("--disable-lpips", action="store_true", help="Disable LPIPS even if installed.")
    parser.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision when CUDA is available.")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No teacher rows found in {path}")
    return rows


def _group_rows_by_base(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("baseId", row.get("sampleId", "sample"))), []).append(row)
    return list(groups.values())


def _split_groups(groups: list[list[dict[str, Any]]], val_split: float, seed: int) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if not groups:
        return [], []
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * max(0.0, min(0.5, val_split)))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _resize_and_pad(image: Image.Image, size: int, *, fill: int | tuple[int, int, int], resample: int) -> Image.Image:
    width, height = image.size
    if max(width, height) == size and width == height:
        return image.copy()
    scale = float(size) / float(max(width, height))
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=resample,
    )
    canvas = Image.new(image.mode, (size, size), color=fill)
    offset = ((size - resized.size[0]) // 2, (size - resized.size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def _load_rgb_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = _resize_and_pad(image, size, fill=(127, 127, 127), resample=Image.LANCZOS)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return tensor


def _load_mask_tensor(path: Path, size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = _resize_and_pad(mask, size, fill=0, resample=Image.NEAREST)
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0) / 255.0
    return (tensor >= 0.5).float()


class TeacherDataset(Dataset):
    def __init__(
        self,
        groups: list[list[dict[str, Any]]],
        *,
        image_size: int,
        sampling_mode: str,
        seed: int,
        family_balance_alpha: float = 0.0,
    ) -> None:
        self.groups = groups
        self.image_size = image_size
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.flat_rows = [row for group in groups for row in group]
        self.family_balance_alpha = max(0.0, float(family_balance_alpha))
        family_counts: dict[str, int] = {}
        for row in self.flat_rows:
            family = str(row.get("promptFamily", "general"))
            family_counts[family] = family_counts.get(family, 0) + 1
        self.family_weights = {
            family: (1.0 / float(count)) ** self.family_balance_alpha
            for family, count in family_counts.items()
        }

    def __len__(self) -> int:
        if self.sampling_mode == "grouped_prompts":
            return len(self.groups)
        return len(self.flat_rows)

    def _select_row(self, index: int) -> dict[str, Any]:
        if self.sampling_mode == "grouped_prompts":
            group = self.groups[index]
            if self.family_balance_alpha > 0:
                weights = torch.tensor(
                    [
                        self.family_weights.get(str(row.get("promptFamily", "general")), 1.0)
                        for row in group
                    ],
                    dtype=torch.float32,
                )
                weights = weights / weights.sum().clamp_min(1e-6)
                choice = torch.multinomial(weights, num_samples=1).item()
            else:
                choice = torch.randint(len(group), size=(1,)).item()
            return group[choice]
        return self.flat_rows[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._select_row(index)
        image_path = Path(row["image"]).expanduser().resolve()
        mask_path = Path(row["mask"]).expanduser().resolve()
        protected_path = Path(row["output"]["protectedImage"]).expanduser().resolve()

        source = _load_rgb_tensor(image_path, self.image_size)
        mask = _load_mask_tensor(mask_path, self.image_size)
        protected = _load_rgb_tensor(protected_path, self.image_size)
        teacher_delta = protected - source

        return {
            "source": source,
            "mask": mask,
            "teacher": protected,
            "teacher_delta": teacher_delta,
            "base_id": str(row.get("baseId", row.get("sampleId", "sample"))),
            "sample_id": str(row.get("sampleId", "sample")),
            "prompt": str(row.get("prompt", "")),
            "prompt_family": str(row.get("promptFamily", "general")),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.down(features)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ProtectorUNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.in_block = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels)
        self.out_block = nn.Sequential(
            ConvBlock(base_channels + base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.in_block(x)
        skip1, down1 = self.down1(stem)
        skip2, down2 = self.down2(down1)
        bottleneck = self.bottleneck(down2)
        up2 = self.up2(bottleneck, skip2)
        up1 = self.up1(up2, skip1)
        merged = torch.cat([up1, stem], dim=1)
        return self.out_block(merged)


def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
    tv_y = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
    tv_x = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
    return tv_x + tv_y


def _build_face_proxy_map(image_tensor: torch.Tensor, protected_mask: torch.Tensor) -> torch.Tensor:
    red = image_tensor[:, 0:1]
    green = image_tensor[:, 1:2]
    blue = image_tensor[:, 2:3]
    max_rgb = image_tensor.amax(dim=1, keepdim=True)
    min_rgb = image_tensor.amin(dim=1, keepdim=True)
    skin_map = (
        (red > 0.35)
        & (green > 0.2)
        & (blue > 0.15)
        & ((max_rgb - min_rgb) > 0.08)
        & (red > green)
        & (red > blue)
    ).to(dtype=torch.float32)
    skin_map = normalize_map(F.avg_pool2d(skin_map, kernel_size=11, stride=1, padding=5))
    face_proxy_map = normalize_map(F.max_pool2d(skin_map, kernel_size=21, stride=1, padding=10))
    return normalize_map(face_proxy_map * protected_mask)


def _build_visibility_weight_map(
    image_tensor: torch.Tensor,
    protected_mask: torch.Tensor,
    face_mask: torch.Tensor,
) -> torch.Tensor:
    gray = rgb_to_luma(image_tensor.to(dtype=torch.float32))
    edge_strength = normalize_map(sobel_magnitude(gray))
    low_texture = (1.0 - edge_strength).clamp(0.0, 1.0)
    face_priority = normalize_map(face_mask)
    smooth_protected = normalize_map(low_texture * protected_mask)
    visibility = (0.7 * face_priority + 0.3 * smooth_protected) * protected_mask
    return visibility.clamp(0.0, 1.0)


def _extract_face_crops(
    image_tensor: torch.Tensor,
    face_mask: torch.Tensor,
    *,
    crop_size: int = 160,
) -> tuple[torch.Tensor, torch.Tensor]:
    crops: list[torch.Tensor] = []
    valid_flags: list[torch.Tensor] = []
    batch_size, _, height, width = image_tensor.shape
    for batch_index in range(batch_size):
        mask = face_mask[batch_index : batch_index + 1]
        binary = mask[0, 0] > 0.15
        if not torch.any(binary):
            crops.append(torch.zeros((3, crop_size, crop_size), device=image_tensor.device, dtype=image_tensor.dtype))
            valid_flags.append(torch.zeros((), device=image_tensor.device, dtype=torch.float32))
            continue
        ys, xs = torch.where(binary)
        y0 = int(ys.min().item())
        y1 = int(ys.max().item()) + 1
        x0 = int(xs.min().item())
        x1 = int(xs.max().item()) + 1
        pad_y = max(2, int(round((y1 - y0) * 0.15)))
        pad_x = max(2, int(round((x1 - x0) * 0.15)))
        y0 = max(0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
        crop = image_tensor[batch_index : batch_index + 1, :, y0:y1, x0:x1]
        crop = F.interpolate(crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(crop[0])
        valid_flags.append(torch.ones((), device=image_tensor.device, dtype=torch.float32))
    return torch.stack(crops, dim=0), torch.stack(valid_flags, dim=0)


def _maybe_build_face_id_model(backend: str, device: torch.device) -> nn.Module | None:
    if backend == "none":
        return None
    if backend in {"auto", "facenet"} and InceptionResnetV1 is not None:
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
    return None


def _compute_real_face_id_similarity(
    model: nn.Module | None,
    source_image: torch.Tensor,
    compare_image: torch.Tensor,
    face_mask: torch.Tensor,
) -> torch.Tensor:
    if model is None:
        return source_image.new_tensor(0.0)
    source_crops, valid = _extract_face_crops(source_image, face_mask)
    compare_crops, _ = _extract_face_crops(compare_image, face_mask)
    if float(valid.sum().item()) <= 0:
        return source_image.new_tensor(0.0)
    source_crops = (source_crops - 0.5) / 0.5
    compare_crops = (compare_crops - 0.5) / 0.5
    source_embeds = F.normalize(model(source_crops), dim=1)
    compare_embeds = F.normalize(model(compare_crops), dim=1)
    similarities = F.cosine_similarity(source_embeds, compare_embeds, dim=1)
    return (similarities * valid).sum() / valid.sum().clamp_min(1.0)


def _maybe_build_lpips(disabled: bool, device: torch.device) -> nn.Module | None:
    if disabled or lpips is None:
        return None
    # LPIPS still instantiates torchvision VGG via the deprecated pretrained flag.
    # Silence that third-party warning until the upstream package updates.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
            category=UserWarning,
        )
        model = lpips.LPIPS(net="vgg")
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _make_drift_config() -> ImmunizationConfig:
    return ImmunizationConfig(
        compression_jitter_strength=0.04,
        subpixel_jitter=0.2,
        frequency_noise_strength=0.0025,
        frequency_band_low=0.08,
        frequency_band_high=0.28,
    )


def _build_purification_view(image_tensor: torch.Tensor, strength: float) -> torch.Tensor:
    strength = max(0.0, float(strength))
    if strength <= 0.0:
        return image_tensor
    purified = apply_light_compression_proxy(image_tensor, min(0.12, 0.02 + 0.9 * strength))
    purified = apply_updown_scale_proxy(purified, scale=max(0.6, 1.0 - 0.85 * strength))
    blur = F.avg_pool2d(purified, kernel_size=3, stride=1, padding=1)
    purified = purified.lerp(blur, min(0.35, 1.2 * strength))
    purified = apply_sharpen_proxy(purified, strength=min(0.12, 0.45 * strength))
    return purified.clamp(0.0, 1.0)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_epsilon: float,
    lpips_model: nn.Module | None,
    face_id_model: nn.Module | None,
    drift_config: ImmunizationConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    if loader is None:
        return {}

    model.eval()
    metrics = {
        "loss": 0.0,
        "teacher_image": 0.0,
        "teacher_delta": 0.0,
        "fidelity_l1": 0.0,
        "lpips": 0.0,
        "face_identity": 0.0,
        "face_drift": 0.0,
        "real_face_id": 0.0,
        "real_face_id_drift": 0.0,
        "purification": 0.0,
        "purification_face": 0.0,
        "selective_visibility": 0.0,
        "visual_tv": 0.0,
        "delta_l2": 0.0,
    }
    steps = 0
    with torch.no_grad():
        for batch in loader:
            steps += 1
            source = batch["source"].to(device=device)
            mask = batch["mask"].to(device=device)
            teacher = batch["teacher"].to(device=device)
            teacher_delta = batch["teacher_delta"].to(device=device)
            protected_mask = (1.0 - mask).clamp(0, 1)
            input_tensor = torch.cat([source, mask], dim=1)

            pred_delta = max_epsilon * torch.tanh(model(input_tensor))
            pred_protected = (source + pred_delta).clamp(0.0, 1.0)

            teacher_image_loss = F.l1_loss(pred_protected, teacher)
            teacher_delta_loss = F.l1_loss(pred_delta * protected_mask, teacher_delta * protected_mask)
            fidelity_l1_loss = F.l1_loss(pred_protected, source)
            lpips_loss = source.new_tensor(0.0)
            if lpips_model is not None:
                lpips_loss = lpips_model(pred_protected * 2.0 - 1.0, source * 2.0 - 1.0).mean()

            face_mask = _build_face_proxy_map(source, protected_mask)
            face_identity_loss = source.new_tensor(0.0)
            face_drift_loss = source.new_tensor(0.0)
            if float(face_mask.mean().item()) > 1e-5:
                source_face_descriptor = compute_masked_descriptor(source, face_mask)
                pred_face_descriptor = compute_masked_descriptor(pred_protected, face_mask)
                face_identity_loss = descriptor_similarity(pred_face_descriptor, source_face_descriptor)

                drift_view = build_identity_drift_view(pred_protected, drift_config)
                drift_face_descriptor = compute_masked_descriptor(drift_view, face_mask)
                face_drift_loss = descriptor_similarity(drift_face_descriptor, source_face_descriptor)

            real_face_id_loss = _compute_real_face_id_similarity(
                face_id_model,
                source,
                pred_protected,
                face_mask,
            )
            purified_pred = _build_purification_view(pred_protected, args.purification_strength)
            purified_teacher = _build_purification_view(teacher, args.purification_strength)
            purification_loss = F.l1_loss(purified_pred, purified_teacher)
            purified_face_loss = source.new_tensor(0.0)
            if float(face_mask.mean().item()) > 1e-5:
                purified_face_loss = descriptor_similarity(
                    compute_masked_descriptor(purified_pred, face_mask),
                    compute_masked_descriptor(source, face_mask),
                )

            real_face_id_drift_loss = _compute_real_face_id_similarity(
                face_id_model,
                source,
                purified_pred,
                face_mask,
            )

            masked_delta = pred_delta * protected_mask
            visibility_map = _build_visibility_weight_map(source, protected_mask, face_mask)
            selective_visibility_loss = (masked_delta.abs() * visibility_map).mean()
            visual_tv_loss = _total_variation(masked_delta)
            delta_l2_loss = masked_delta.square().mean()

            total_loss = (
                args.teacher_image_weight * teacher_image_loss
                + args.teacher_delta_weight * teacher_delta_loss
                + args.fidelity_l1_weight * fidelity_l1_loss
                + args.lpips_weight * lpips_loss
                + args.face_identity_weight * face_identity_loss
                + args.face_drift_weight * face_drift_loss
                + args.real_face_id_weight * real_face_id_loss
                + args.real_face_id_drift_weight * real_face_id_drift_loss
                + args.purification_consistency_weight * purification_loss
                + args.purification_face_weight * purified_face_loss
                + args.selective_visibility_weight * selective_visibility_loss
                + args.visual_tv_weight * visual_tv_loss
                + args.delta_l2_weight * delta_l2_loss
            )

            metrics["loss"] += float(total_loss.item())
            metrics["teacher_image"] += float(teacher_image_loss.item())
            metrics["teacher_delta"] += float(teacher_delta_loss.item())
            metrics["fidelity_l1"] += float(fidelity_l1_loss.item())
            metrics["lpips"] += float(lpips_loss.item())
            metrics["face_identity"] += float(face_identity_loss.item())
            metrics["face_drift"] += float(face_drift_loss.item())
            metrics["real_face_id"] += float(real_face_id_loss.item())
            metrics["real_face_id_drift"] += float(real_face_id_drift_loss.item())
            metrics["purification"] += float(purification_loss.item())
            metrics["purification_face"] += float(purified_face_loss.item())
            metrics["selective_visibility"] += float(selective_visibility_loss.item())
            metrics["visual_tv"] += float(visual_tv_loss.item())
            metrics["delta_l2"] += float(delta_l2_loss.item())

    if steps == 0:
        return {}
    return {key: value / steps for key, value in metrics.items()}


def main() -> int:
    args = _parse_args()
    _seed_everything(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    rows = _load_manifest_rows(manifest_path, limit=args.limit)
    groups = _group_rows_by_base(rows)
    train_groups, val_groups = _split_groups(groups, args.val_split, args.seed)
    if not train_groups:
        raise RuntimeError("Training split is empty. Provide more teacher rows or reduce --val-split.")

    train_dataset = TeacherDataset(
        train_groups,
        image_size=args.image_size,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        family_balance_alpha=args.family_balance_alpha,
    )
    val_dataset = None
    if val_groups:
        val_dataset = TeacherDataset(
            val_groups,
            image_size=args.image_size,
            sampling_mode="flat",
            seed=args.seed + 1,
            family_balance_alpha=0.0,
        )

    train_sampler = None
    train_shuffle = args.sampling_mode != "grouped_prompts"
    if args.sampling_mode == "flat" and args.family_balance_alpha > 0:
        sample_weights = [
            train_dataset.family_weights.get(
                str(row.get("promptFamily", "general")),
                1.0,
            )
            for row in train_dataset.flat_rows
        ]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtectorUNet(base_channels=args.base_channels).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    lpips_model = _maybe_build_lpips(args.disable_lpips, device)
    face_id_model = _maybe_build_face_id_model(args.face_id_backend, device)
    effective_face_id_backend = args.face_id_backend if face_id_model is not None else "none"
    drift_config = _make_drift_config()

    best_val = math.inf
    with history_path.open("w", encoding="utf-8") as handle:
        handle.write("")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "teacher_image": 0.0,
            "teacher_delta": 0.0,
            "fidelity_l1": 0.0,
            "lpips": 0.0,
            "face_identity": 0.0,
            "face_drift": 0.0,
            "real_face_id": 0.0,
            "real_face_id_drift": 0.0,
            "purification": 0.0,
            "purification_face": 0.0,
            "selective_visibility": 0.0,
            "visual_tv": 0.0,
            "delta_l2": 0.0,
        }
        steps = 0

        for batch in train_loader:
            steps += 1
            source = batch["source"].to(device=device, non_blocking=True)
            mask = batch["mask"].to(device=device, non_blocking=True)
            teacher = batch["teacher"].to(device=device, non_blocking=True)
            teacher_delta = batch["teacher_delta"].to(device=device, non_blocking=True)
            protected_mask = (1.0 - mask).clamp(0, 1)
            input_tensor = torch.cat([source, mask], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred_delta = args.max_epsilon * torch.tanh(model(input_tensor))
                pred_protected = (source + pred_delta).clamp(0.0, 1.0)

                teacher_image_loss = F.l1_loss(pred_protected, teacher)
                teacher_delta_loss = F.l1_loss(pred_delta * protected_mask, teacher_delta * protected_mask)
                fidelity_l1_loss = F.l1_loss(pred_protected, source)
                lpips_loss = source.new_tensor(0.0)
                if lpips_model is not None:
                    lpips_loss = lpips_model(pred_protected * 2.0 - 1.0, source * 2.0 - 1.0).mean()

                face_mask = _build_face_proxy_map(source, protected_mask)
                face_identity_loss = source.new_tensor(0.0)
                face_drift_loss = source.new_tensor(0.0)
                if float(face_mask.mean().item()) > 1e-5:
                    source_face_descriptor = compute_masked_descriptor(source, face_mask)
                    pred_face_descriptor = compute_masked_descriptor(pred_protected, face_mask)
                    face_identity_loss = descriptor_similarity(pred_face_descriptor, source_face_descriptor)

                    drift_view = build_identity_drift_view(pred_protected, drift_config)
                    drift_face_descriptor = compute_masked_descriptor(drift_view, face_mask)
                    face_drift_loss = descriptor_similarity(drift_face_descriptor, source_face_descriptor)

                real_face_id_loss = _compute_real_face_id_similarity(
                    face_id_model,
                    source,
                    pred_protected,
                    face_mask,
                )
                purified_pred = _build_purification_view(pred_protected, args.purification_strength)
                purified_teacher = _build_purification_view(teacher, args.purification_strength)
                purification_loss = F.l1_loss(purified_pred, purified_teacher)
                purified_face_loss = source.new_tensor(0.0)
                if float(face_mask.mean().item()) > 1e-5:
                    purified_face_loss = descriptor_similarity(
                        compute_masked_descriptor(purified_pred, face_mask),
                        compute_masked_descriptor(source, face_mask),
                    )

                real_face_id_drift_loss = _compute_real_face_id_similarity(
                    face_id_model,
                    source,
                    purified_pred,
                    face_mask,
                )

                masked_delta = pred_delta * protected_mask
                visibility_map = _build_visibility_weight_map(source, protected_mask, face_mask)
                selective_visibility_loss = (masked_delta.abs() * visibility_map).mean()
                visual_tv_loss = _total_variation(masked_delta)
                delta_l2_loss = masked_delta.square().mean()

                total_loss = (
                    args.teacher_image_weight * teacher_image_loss
                    + args.teacher_delta_weight * teacher_delta_loss
                    + args.fidelity_l1_weight * fidelity_l1_loss
                    + args.lpips_weight * lpips_loss
                    + args.face_identity_weight * face_identity_loss
                    + args.face_drift_weight * face_drift_loss
                    + args.real_face_id_weight * real_face_id_loss
                    + args.real_face_id_drift_weight * real_face_id_drift_loss
                    + args.purification_consistency_weight * purification_loss
                    + args.purification_face_weight * purified_face_loss
                    + args.selective_visibility_weight * selective_visibility_loss
                    + args.visual_tv_weight * visual_tv_loss
                    + args.delta_l2_weight * delta_l2_loss
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_metrics["loss"] += float(total_loss.item())
            epoch_metrics["teacher_image"] += float(teacher_image_loss.item())
            epoch_metrics["teacher_delta"] += float(teacher_delta_loss.item())
            epoch_metrics["fidelity_l1"] += float(fidelity_l1_loss.item())
            epoch_metrics["lpips"] += float(lpips_loss.item())
            epoch_metrics["face_identity"] += float(face_identity_loss.item())
            epoch_metrics["face_drift"] += float(face_drift_loss.item())
            epoch_metrics["real_face_id"] += float(real_face_id_loss.item())
            epoch_metrics["real_face_id_drift"] += float(real_face_id_drift_loss.item())
            epoch_metrics["purification"] += float(purification_loss.item())
            epoch_metrics["purification_face"] += float(purified_face_loss.item())
            epoch_metrics["selective_visibility"] += float(selective_visibility_loss.item())
            epoch_metrics["visual_tv"] += float(visual_tv_loss.item())
            epoch_metrics["delta_l2"] += float(delta_l2_loss.item())

        averaged_train = {key: value / max(1, steps) for key, value in epoch_metrics.items()}
        averaged_val = _evaluate(
            model,
            val_loader,
            device=device,
            max_epsilon=args.max_epsilon,
            lpips_model=lpips_model,
            face_id_model=face_id_model,
            drift_config=drift_config,
            args=args,
        ) if val_loader is not None else {}

        log_row = {
            "epoch": epoch,
            "train": averaged_train,
            "val": averaged_val,
            "config": {
                "imageSize": args.image_size,
                "samplingMode": args.sampling_mode,
                "familyBalanceAlpha": args.family_balance_alpha,
                "maxEpsilon": args.max_epsilon,
                "faceIdBackend": effective_face_id_backend,
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": averaged_train,
            "val_metrics": averaged_val,
            "args": vars(args),
        }
        if epoch % max(1, args.save_every) == 0:
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, checkpoints_dir / "latest.pt")

        comparison_value = averaged_val.get("loss", averaged_train["loss"])
        if comparison_value < best_val:
            best_val = comparison_value
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        train_suffix = ", ".join(f"{key}={value:.4f}" for key, value in averaged_train.items())
        if averaged_val:
            val_suffix = ", ".join(f"{key}={value:.4f}" for key, value in averaged_val.items())
            print(f"[epoch {epoch}] train: {train_suffix} | val: {val_suffix}", flush=True)
        else:
            print(f"[epoch {epoch}] train: {train_suffix}", flush=True)

    summary = {
        "epochs": args.epochs,
        "trainGroups": len(train_groups),
        "valGroups": len(val_groups),
        "outputDir": str(output_dir),
        "faceIdBackend": effective_face_id_backend,
        "checkpoints": {
            "best": str(checkpoints_dir / "best.pt"),
            "latest": str(checkpoints_dir / "latest.pt"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
