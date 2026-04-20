#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.immunization import compute_masked_descriptor, descriptor_similarity  # noqa: E402
from backend.vision_surrogates import (  # noqa: E402
    VisionSurrogate,
    compute_vision_surrogate_embedding,
    maybe_load_vision_surrogate,
)
from scripts.train_protector import (  # noqa: E402
    _build_face_proxy_map,
    _compute_real_face_id_similarity,
    _load_mask_tensor,
    _load_rgb_tensor,
    _maybe_build_face_id_model,
    _maybe_build_lpips,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate external-eval templates or score manual DeeVid/NB2 edit results.",
    )
    parser.add_argument("--teacher-manifest", type=Path, default=None, help="Teacher manifest.jsonl to convert into an external-eval CSV template.")
    parser.add_argument("--template-out", type=Path, default=None, help="Path to write the CSV template.")
    parser.add_argument("--cases-csv", type=Path, default=None, help="Filled CSV of external edit cases to score.")
    parser.add_argument("--report-out", type=Path, default=None, help="Output CSV for computed metrics.")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for metric evaluation.")
    parser.add_argument("--face-id-backend", choices=("auto", "none", "facenet"), default="auto")
    parser.add_argument("--vision-metrics", choices=("auto", "none"), default="auto")
    parser.add_argument("--disable-lpips", action="store_true", help="Disable LPIPS even if installed.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _write_template(teacher_manifest: Path, template_out: Path) -> None:
    rows = _load_jsonl(teacher_manifest)
    fieldnames = [
        "case_id",
        "base_id",
        "tool",
        "prompt",
        "prompt_family",
        "source",
        "protected",
        "unprotected_edit",
        "protected_edit",
        "mask",
        "notes",
    ]
    template_out.parent.mkdir(parents=True, exist_ok=True)
    with template_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = row.get("output", {}) if isinstance(row, dict) else {}
            writer.writerow(
                {
                    "case_id": row.get("sampleId", ""),
                    "base_id": row.get("baseId", ""),
                    "tool": "",
                    "prompt": row.get("prompt", ""),
                    "prompt_family": row.get("promptFamily", "general"),
                    "source": row.get("image", ""),
                    "protected": output.get("protectedImage", "") if isinstance(output, dict) else "",
                    "unprotected_edit": "",
                    "protected_edit": "",
                    "mask": row.get("mask", ""),
                    "notes": "",
                }
            )


def _load_cases_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _compute_vision_similarity(
    surrogate: VisionSurrogate | None,
    source_image: torch.Tensor,
    compare_image: torch.Tensor,
    spatial_mask: torch.Tensor,
) -> torch.Tensor:
    if surrogate is None:
        return source_image.new_tensor(0.0)
    source_embedding = compute_vision_surrogate_embedding(surrogate, source_image, spatial_mask)
    compare_embedding = compute_vision_surrogate_embedding(surrogate, compare_image, spatial_mask)
    if source_embedding is None or compare_embedding is None:
        return source_image.new_tensor(0.0)
    return F.cosine_similarity(source_embedding, compare_embedding, dim=1).mean()


def _score_cases(
    rows: list[dict[str, str]],
    *,
    image_size: int,
    face_id_backend: str,
    vision_metrics: str,
    disable_lpips: bool,
) -> list[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = _maybe_build_lpips(disable_lpips, device)
    face_id_model = _maybe_build_face_id_model(face_id_backend, device)
    clip_surrogate = maybe_load_vision_surrogate("clip", device) if vision_metrics == "auto" else None
    dino_surrogate = maybe_load_vision_surrogate("dino", device) if vision_metrics == "auto" else None
    scored: list[dict[str, object]] = []
    for row in rows:
        source = _load_rgb_tensor(Path(row["source"]).expanduser().resolve(), image_size).unsqueeze(0).to(device)
        protected = _load_rgb_tensor(Path(row["protected"]).expanduser().resolve(), image_size).unsqueeze(0).to(device)
        unprotected_edit = _load_rgb_tensor(Path(row["unprotected_edit"]).expanduser().resolve(), image_size).unsqueeze(0).to(device)
        protected_edit = _load_rgb_tensor(Path(row["protected_edit"]).expanduser().resolve(), image_size).unsqueeze(0).to(device)

        if row.get("mask"):
            mask = _load_mask_tensor(Path(row["mask"]).expanduser().resolve(), image_size).unsqueeze(0).to(device)
            protected_mask = (1.0 - mask).clamp(0.0, 1.0)
        else:
            protected_mask = torch.ones((1, 1, source.shape[-2], source.shape[-1]), device=device, dtype=source.dtype)

        face_mask = _build_face_proxy_map(source, protected_mask)
        source_desc = compute_masked_descriptor(source, protected_mask)
        unprotected_desc = compute_masked_descriptor(unprotected_edit, protected_mask)
        protected_desc = compute_masked_descriptor(protected_edit, protected_mask)

        protected_cost_l1 = F.l1_loss(protected, source).item()
        unprotected_edit_l1 = F.l1_loss(unprotected_edit, source).item()
        protected_edit_l1 = F.l1_loss(protected_edit, source).item()
        if lpips_model is not None:
            protected_cost_lpips = float(lpips_model(protected * 2.0 - 1.0, source * 2.0 - 1.0).mean().item())
            unprotected_edit_lpips = float(lpips_model(unprotected_edit * 2.0 - 1.0, source * 2.0 - 1.0).mean().item())
            protected_edit_lpips = float(lpips_model(protected_edit * 2.0 - 1.0, source * 2.0 - 1.0).mean().item())
        else:
            protected_cost_lpips = 0.0
            unprotected_edit_lpips = 0.0
            protected_edit_lpips = 0.0

        generic_similarity_unprotected = float(descriptor_similarity(source_desc, unprotected_desc).item())
        generic_similarity_protected = float(descriptor_similarity(source_desc, protected_desc).item())
        face_similarity_unprotected = 0.0
        face_similarity_protected = 0.0
        if float(face_mask.mean().item()) > 1e-5:
            face_similarity_unprotected = float(
                descriptor_similarity(
                    compute_masked_descriptor(source, face_mask),
                    compute_masked_descriptor(unprotected_edit, face_mask),
                ).item()
            )
            face_similarity_protected = float(
                descriptor_similarity(
                    compute_masked_descriptor(source, face_mask),
                    compute_masked_descriptor(protected_edit, face_mask),
                ).item()
            )

        real_face_similarity_unprotected = float(
            _compute_real_face_id_similarity(face_id_model, source, unprotected_edit, face_mask).item()
        )
        real_face_similarity_protected = float(
            _compute_real_face_id_similarity(face_id_model, source, protected_edit, face_mask).item()
        )
        clip_similarity_unprotected = float(
            _compute_vision_similarity(clip_surrogate, source, unprotected_edit, protected_mask).item()
        )
        clip_similarity_protected = float(
            _compute_vision_similarity(clip_surrogate, source, protected_edit, protected_mask).item()
        )
        dino_similarity_unprotected = float(
            _compute_vision_similarity(dino_surrogate, source, unprotected_edit, protected_mask).item()
        )
        dino_similarity_protected = float(
            _compute_vision_similarity(dino_surrogate, source, protected_edit, protected_mask).item()
        )

        scored.append(
            {
                **row,
                "protected_cost_l1": protected_cost_l1,
                "protected_cost_lpips": protected_cost_lpips,
                "unprotected_edit_l1": unprotected_edit_l1,
                "protected_edit_l1": protected_edit_l1,
                "unprotected_edit_lpips": unprotected_edit_lpips,
                "protected_edit_lpips": protected_edit_lpips,
                "edit_attenuation_l1": unprotected_edit_l1 - protected_edit_l1,
                "edit_attenuation_lpips": unprotected_edit_lpips - protected_edit_lpips,
                "generic_similarity_unprotected": generic_similarity_unprotected,
                "generic_similarity_protected": generic_similarity_protected,
                "face_similarity_unprotected": face_similarity_unprotected,
                "face_similarity_protected": face_similarity_protected,
                "real_face_similarity_unprotected": real_face_similarity_unprotected,
                "real_face_similarity_protected": real_face_similarity_protected,
                "clip_similarity_unprotected": clip_similarity_unprotected,
                "clip_similarity_protected": clip_similarity_protected,
                "clip_similarity_gain": clip_similarity_protected - clip_similarity_unprotected,
                "dino_similarity_unprotected": dino_similarity_unprotected,
                "dino_similarity_protected": dino_similarity_protected,
                "dino_similarity_gain": dino_similarity_protected - dino_similarity_unprotected,
            }
        )
    return scored


def _write_report(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    if args.teacher_manifest and args.template_out:
        _write_template(args.teacher_manifest.expanduser().resolve(), args.template_out.expanduser().resolve())
        print(f"Wrote template to {args.template_out}", flush=True)
        return 0
    if args.cases_csv and args.report_out:
        rows = _load_cases_csv(args.cases_csv.expanduser().resolve())
        scored = _score_cases(
            rows,
            image_size=args.image_size,
            face_id_backend=args.face_id_backend,
            vision_metrics=args.vision_metrics,
            disable_lpips=args.disable_lpips,
        )
        _write_report(scored, args.report_out.expanduser().resolve())
        print(f"Wrote report to {args.report_out}", flush=True)
        return 0
    raise SystemExit("Provide either --teacher-manifest with --template-out, or --cases-csv with --report-out.")


if __name__ == "__main__":
    raise SystemExit(main())
