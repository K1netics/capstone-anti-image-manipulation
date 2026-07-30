#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate person masks for a directory of images using a pretrained "
            "DeepLabV3 segmentation model."
        )
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where generated masks will be written.",
    )
    parser.add_argument(
        "--preview-dir",
        default="",
        help="Optional directory for QA overlay previews.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png,*.webp",
        help="Comma-separated filename globs to scan under --images.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.22,
        help="Person probability threshold before post-processing.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=18,
        help="Base number of 3x3 max-filter passes after resizing to the original size.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.004,
        help=(
            "Extra growth radius as a fraction of the largest image dimension. "
            "Use this to add a little more breathing room around the full subject."
        ),
    )
    parser.add_argument(
        "--close",
        type=int,
        default=2,
        help="Number of close passes (max then min) to fill small holes.",
    )
    parser.add_argument(
        "--final-close",
        type=int,
        default=1,
        help="Additional close passes after dilation to smooth edges and seal small gaps.",
    )
    parser.add_argument(
        "--fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill enclosed holes inside the final binary mask.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum kept component size relative to the low-res mask area.",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.08,
        help="Mark masks with less total coverage than this as review candidates.",
    )
    parser.add_argument(
        "--min-bbox-height-ratio",
        type=float,
        default=0.28,
        help="Mark masks whose bounding box is shorter than this fraction of image height.",
    )
    parser.add_argument(
        "--max-mask-ratio",
        type=float,
        default=0.55,
        help="Mark masks with more total coverage than this as review candidates.",
    )
    parser.add_argument(
        "--review-border-touch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark masks that touch an image border as review candidates.",
    )
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="Keep only the largest connected person component before resizing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks.",
    )
    parser.add_argument(
        "--prefix",
        default="Masked-",
        help="Prefix added to the output filename stem.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def discover_images(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(path for path in files if path.is_file())


def largest_component(mask: np.ndarray, *, min_pixels: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    best_component: list[tuple[int, int]] = []
    best_size = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component: list[tuple[int, int]] = []

            while queue:
                cy, cx = queue.pop()
                component.append((cy, cx))

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component) >= min_pixels and len(component) > best_size:
                best_component = component
                best_size = len(component)

    if not best_component:
        return mask

    filtered = np.zeros_like(mask, dtype=bool)
    for y, x in best_component:
        filtered[y, x] = True
    return filtered


def postprocess_mask(
    mask_small: np.ndarray,
    *,
    output_size: tuple[int, int],
    close_passes: int,
    dilate_passes: int,
    margin_ratio: float,
    final_close_passes: int,
    fill_holes: bool,
) -> Image.Image:
    mask = Image.fromarray((mask_small.astype(np.uint8) * 255), mode="L")
    mask = mask.resize(output_size, resample=Image.NEAREST)

    for _ in range(max(0, close_passes)):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))
        mask = mask.filter(ImageFilter.MinFilter(size=3))

    adaptive_dilate = max(0, int(round(max(output_size) * max(0.0, margin_ratio))))
    total_dilate_passes = max(0, dilate_passes + adaptive_dilate)

    for _ in range(total_dilate_passes):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))

    for _ in range(max(0, final_close_passes)):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))
        mask = mask.filter(ImageFilter.MinFilter(size=3))

    mask_array = np.asarray(mask, dtype=np.uint8) >= 128
    if fill_holes:
        mask_array = fill_mask_holes(mask_array)

    return Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    exterior = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(y: int, x: int) -> None:
        if not mask[y, x] and not exterior[y, x]:
            exterior[y, x] = True
            queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in (
            (cy - 1, cx),
            (cy + 1, cx),
            (cy, cx - 1),
            (cy, cx + 1),
        ):
            if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                queue.append((ny, nx))

    holes = (~mask) & (~exterior)
    return mask | holes


def compute_mask_stats(mask: Image.Image) -> dict[str, Any]:
    mask_array = np.asarray(mask, dtype=np.uint8) >= 128
    height, width = mask_array.shape
    foreground = int(mask_array.sum())
    coverage = float(foreground / mask_array.size)

    if foreground == 0:
        return {
            "coverage": coverage,
            "bboxWidthRatio": 0.0,
            "bboxHeightRatio": 0.0,
            "touchTop": False,
            "touchBottom": False,
            "touchLeft": False,
            "touchRight": False,
        }

    ys, xs = np.where(mask_array)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    return {
        "coverage": coverage,
        "bboxWidthRatio": float((x1 - x0 + 1) / width),
        "bboxHeightRatio": float((y1 - y0 + 1) / height),
        "touchTop": bool(y0 == 0),
        "touchBottom": bool(y1 == height - 1),
        "touchLeft": bool(x0 == 0),
        "touchRight": bool(x1 == width - 1),
    }


def classify_mask(
    stats: dict[str, Any],
    *,
    min_mask_ratio: float,
    min_bbox_height_ratio: float,
    max_mask_ratio: float,
    review_border_touch: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if stats["coverage"] < min_mask_ratio:
        reasons.append("small_mask_area")
    if stats["bboxHeightRatio"] < min_bbox_height_ratio:
        reasons.append("short_subject_bbox")
    if stats["coverage"] > max_mask_ratio:
        reasons.append("large_mask_area")
    if review_border_touch and (
        stats["touchTop"]
        or stats["touchBottom"]
        or stats["touchLeft"]
        or stats["touchRight"]
    ):
        reasons.append("touches_image_border")

    return ("review", reasons) if reasons else ("keep", [])


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(record.get("mask") or "")
        if key:
            records[key] = record
    return records


def build_preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color=(255, 255, 255, 0))
    alpha = mask.point(lambda value: 96 if value > 0 else 0, mode="L")
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay)


def output_name(image_path: Path, *, prefix: str) -> str:
    return f"{prefix}{image_path.stem}.png"


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    preview_dir = Path(args.preview_dir).expanduser().resolve() if args.preview_dir else None

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    patterns = [part.strip() for part in args.pattern.split(",") if part.strip()]
    image_paths = discover_images(images_dir, patterns)
    if not image_paths:
        raise FileNotFoundError(f"No input images matched under {images_dir}")

    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet101(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    categories = list(weights.meta.get("categories", ()))
    if "person" not in categories:
        raise RuntimeError("The selected segmentation weights do not expose a 'person' category.")
    person_index = categories.index("person")

    manifest_path = output_dir / "mask_manifest.jsonl"
    manifest_records = load_manifest(manifest_path)

    print(f"Found {len(image_paths)} images. Using device: {device}", flush=True)

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = output_dir / output_name(image_path, prefix=args.prefix)
        if mask_path.exists() and not args.overwrite:
            print(f"[{index}/{len(image_paths)}] skipping existing {mask_path.name}", flush=True)
            continue

        with Image.open(image_path) as image_fp:
            image = image_fp.convert("RGB")

        with torch.inference_mode():
            tensor = preprocess(image).unsqueeze(0).to(device)
            logits = model(tensor)["out"][0].detach().cpu()
            probs = torch.softmax(logits, dim=0)[person_index].numpy()

        mask_small = probs >= float(args.threshold)

        if args.keep_largest:
            min_pixels = max(8, int(mask_small.size * float(args.min_area_ratio)))
            mask_small = largest_component(mask_small, min_pixels=min_pixels)

        final_mask = postprocess_mask(
            mask_small,
            output_size=image.size,
            close_passes=args.close,
            dilate_passes=args.dilate,
            margin_ratio=args.margin_ratio,
            final_close_passes=args.final_close,
            fill_holes=bool(args.fill_holes),
        )
        final_mask.save(mask_path)

        preview_path = None
        if preview_dir is not None:
            preview_path = preview_dir / output_name(image_path, prefix=args.prefix)
            build_preview(image, final_mask).save(preview_path)

        stats = compute_mask_stats(final_mask)
        status, review_reasons = classify_mask(
            stats,
            min_mask_ratio=float(args.min_mask_ratio),
            min_bbox_height_ratio=float(args.min_bbox_height_ratio),
            max_mask_ratio=float(args.max_mask_ratio),
            review_border_touch=bool(args.review_border_touch),
        )
        record = {
            "image": str(image_path),
            "mask": str(mask_path),
            "preview": str(preview_path) if preview_path is not None else None,
            "coverage": stats["coverage"],
            "bboxWidthRatio": stats["bboxWidthRatio"],
            "bboxHeightRatio": stats["bboxHeightRatio"],
            "touchTop": stats["touchTop"],
            "touchBottom": stats["touchBottom"],
            "touchLeft": stats["touchLeft"],
            "touchRight": stats["touchRight"],
            "status": status,
            "reviewReasons": review_reasons,
            "size": [image.size[0], image.size[1]],
            "threshold": args.threshold,
            "close": args.close,
            "finalClose": args.final_close,
            "dilate": args.dilate,
            "marginRatio": args.margin_ratio,
            "fillHoles": bool(args.fill_holes),
            "keepLargest": bool(args.keep_largest),
        }
        manifest_records[str(mask_path)] = record

        review_suffix = ""
        if review_reasons:
            review_suffix = f" [{status}: {', '.join(review_reasons)}]"
        print(
            f"[{index}/{len(image_paths)}] wrote {mask_path.name} "
            f"(coverage {stats['coverage']:.3f}, bbox_h {stats['bboxHeightRatio']:.3f})"
            f"{review_suffix}",
            flush=True,
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for key in sorted(manifest_records):
            handle.write(json.dumps(manifest_records[key]) + "\n")

    print(f"Done. Masks written to {output_dir}", flush=True)
    if preview_dir is not None:
        print(f"Preview overlays written to {preview_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
