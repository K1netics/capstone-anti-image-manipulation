#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.feedback import Feedback  # noqa: E402
from api.main import (  # noqa: E402
    DEFAULT_IMMUNIZATION_PROFILE,
    DEFAULT_WORKING_RESOLUTION,
    BackendSettingOverrides,
    _append_cuda_memory_feedback,
    _apply_backend_setting_overrides,
    _finalize_output_image,
    _get_pipeline,
    _immunization_config_for_profile,
    _low_memory_immunization_config,
    _prepare_request_images,
    _report_pipeline_runtime_features,
    _reset_cuda_peak_memory_stats,
    _resolve_immunization_profile,
    _resolve_immunization_resolution,
    _resize_for_immunization_fallback,
    _try_immunize_once,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher-protected images for student-model training.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to a JSONL file describing image/mask/prompt samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where teacher samples and manifests will be written.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Optional base directory for relative image/mask paths. Defaults to metadata file parent.",
    )
    parser.add_argument(
        "--working-resolution",
        choices=("512", "1024", "original"),
        default=DEFAULT_WORKING_RESOLUTION,
        help="Working resolution passed through the upgraded preprocessing path.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_IMMUNIZATION_PROFILE,
        help="Immunization profile name, e.g. stable_diffusion or nano_banana_2.",
    )
    parser.add_argument(
        "--defense-canvas",
        default="768",
        choices=("profile_default", "512", "640", "768", "1024", "working"),
        help="Defense canvas override.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for the local proxy and final edit path assumptions.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Inference steps used by the SDXL proxy during immunization.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=40,
        help="PGD iteration override.",
    )
    parser.add_argument(
        "--eot-samples",
        type=int,
        default=3,
        help="EOT sample override.",
    )
    parser.add_argument(
        "--prompt-variants",
        type=int,
        default=6,
        help="Max prompt variants override.",
    )
    parser.add_argument(
        "--denoiser-strength",
        type=float,
        default=0.26,
        help="Denoiser strength override.",
    )
    parser.add_argument(
        "--reference-confusion",
        type=float,
        default=0.12,
        help="Reference confusion strength override.",
    )
    parser.add_argument(
        "--identity-drift",
        type=float,
        default=0.10,
        help="Identity drift strength override.",
    )
    parser.add_argument(
        "--semantic-boundary",
        type=float,
        default=0.16,
        help="Semantic boundary strength override.",
    )
    parser.add_argument(
        "--watermark",
        type=float,
        default=0.06,
        help="Watermark strength override.",
    )
    parser.add_argument(
        "--global-anchor",
        type=float,
        default=0.08,
        help="Global anchor strength override.",
    )
    parser.add_argument(
        "--early-step-bias",
        type=float,
        default=1.6,
        help="Bias denoiser supervision toward earlier diffusion timesteps.",
    )
    parser.add_argument(
        "--mask-augment-strength",
        type=float,
        default=0.2,
        help="Strength of in-optimizer mask augmentation used by the teacher.",
    )
    parser.add_argument(
        "--mask-augment-count",
        type=int,
        default=2,
        help="Number of in-optimizer mask views used by the teacher.",
    )
    parser.add_argument(
        "--portrait-face-identity",
        type=float,
        default=0.08,
        help="Portrait-specific face identity confusion strength.",
    )
    parser.add_argument(
        "--mask-variant-count",
        type=int,
        default=0,
        help="How many dataset-level mask variants to generate per source mask.",
    )
    parser.add_argument(
        "--mask-variant-strength",
        type=float,
        default=0.0,
        help="Strength of dataset-level mask-variant augmentation.",
    )
    parser.add_argument(
        "--allow-low-memory-fallback",
        action="store_true",
        help="Allow low-memory defense fallback instead of failing closed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have protected outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompt-expanded samples to run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first sample failure.",
    )
    return parser.parse_args()


def _load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {index} of {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata line {index} must be a JSON object.")
        payload["_line"] = index
        rows.append(payload)
    return rows


def _expand_prompts(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("prompts"), list):
        prompts = [str(value).strip() for value in row["prompts"] if str(value).strip()]
        if prompts:
            return prompts
    prompt = str(row.get("prompt", "")).strip()
    if prompt:
        return [prompt]
    raise ValueError("Each metadata row needs either 'prompt' or 'prompts'.")


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _row_id(row: dict[str, Any], fallback_index: int) -> str:
    value = str(row.get("id", "")).strip()
    if value:
        return value
    image_stem = Path(str(row.get("image", f"sample_{fallback_index:05d}"))).stem
    return image_stem or f"sample_{fallback_index:05d}"


def _sample_slug(base_id: str, prompt: str, prompt_index: int, mask_variant: str = "original") -> str:
    digest = hashlib.sha1(f"{mask_variant}::{prompt}".encode("utf-8")).hexdigest()[:8]
    if mask_variant == "original":
        return f"{base_id}__p{prompt_index:02d}__{digest}"
    return f"{base_id}__m{mask_variant}__p{prompt_index:02d}__{digest}"


def _mask_to_png_bytes(mask_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _shift_mask(mask_image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    shifted = ImageChops.offset(mask_image, shift_x, shift_y)
    width, height = mask_image.size
    if shift_x > 0:
        shifted.paste(0, (0, 0, shift_x, height))
    elif shift_x < 0:
        shifted.paste(0, (width + shift_x, 0, width, height))
    if shift_y > 0:
        shifted.paste(0, (0, 0, width, shift_y))
    elif shift_y < 0:
        shifted.paste(0, (0, height + shift_y, width, height))
    return shifted


def _build_mask_variants(mask_image: Image.Image, count: int, strength: float) -> list[tuple[str, Image.Image]]:
    base_mask = mask_image.convert("L")
    variants = [("original", base_mask)]
    if count <= 0 or strength <= 0:
        return variants

    max_dim = max(base_mask.size)
    radius = max(1, int(round(max_dim * min(0.03, 0.004 + 0.014 * float(strength)))))
    kernel_size = radius * 2 + 1
    shift = max(1, int(round(max_dim * min(0.015, 0.002 + 0.008 * float(strength)))))
    candidates = [
        ("dilate", base_mask.filter(ImageFilter.MaxFilter(kernel_size))),
        ("erode", base_mask.filter(ImageFilter.MinFilter(kernel_size))),
        ("shift_right", _shift_mask(base_mask, shift, 0)),
        ("shift_left", _shift_mask(base_mask, -shift, 0)),
        ("shift_down", _shift_mask(base_mask, 0, shift)),
        ("shift_up", _shift_mask(base_mask, 0, -shift)),
    ]

    seen = {hashlib.md5(base_mask.tobytes()).hexdigest()}
    for name, variant in candidates:
        normalized = variant.point(lambda value: 255 if value >= 128 else 0, mode="L")
        digest = hashlib.md5(normalized.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        variants.append((name, normalized))
        if len(variants) >= count + 1:
            break
    return variants


def _infer_prompt_family(prompt: str) -> str:
    normalized = prompt.lower()
    if any(token in normalized for token in ("sunglass", "glasses", "eyewear")):
        return "eyewear"
    if any(token in normalized for token in ("shirt", "dress", "jacket", "hoodie", "clothing", "blazer")):
        return "clothing"
    if any(token in normalized for token in ("hair", "hairstyle", "haircut", "hair color")):
        return "hair"
    if any(token in normalized for token in ("logo", "text", "overlay")):
        return "text_logo"
    if any(token in normalized for token in ("background", "scene", "outfit swap", "remove")):
        return "scene_structure"
    return "general"


def _save_delta(source: Image.Image, protected: Image.Image, output_path: Path) -> None:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    protected_array = np.asarray(protected.convert("RGB"), dtype=np.float32) / 255.0
    delta = (protected_array - source_array).astype(np.float16)
    np.save(output_path, delta)


def _save_delta_visualization(source: Image.Image, protected: Image.Image, output_path: Path) -> None:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    protected_array = np.asarray(protected.convert("RGB"), dtype=np.float32) / 255.0
    delta = protected_array - source_array
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs < 1e-6:
        vis = np.full(delta.shape, 127, dtype=np.uint8)
    else:
        vis = np.clip((delta / max_abs) * 127.0 + 127.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(vis, mode="RGB").save(output_path, format="PNG", optimize=True)


def _progress_logger(sample_slug: str):
    last_logged_iteration = {"value": -1}

    def callback(update: dict[str, object]) -> None:
        iteration = int(update.get("iteration", 0) or 0)
        total_iterations = int(update.get("totalIterations", 0) or 0)
        should_log = (
            iteration <= 1
            or iteration == total_iterations
            or (iteration % 5 == 0 and iteration != last_logged_iteration["value"])
        )
        if not should_log:
            return
        last_logged_iteration["value"] = iteration
        metrics = update.get("metrics")
        loss = update.get("loss")
        metric_suffix = ""
        if isinstance(metrics, dict) and metrics:
            interesting = []
            for key in ("target", "denoiser", "reference", "identity", "semantic", "watermark"):
                if isinstance(metrics.get(key), (int, float)):
                    interesting.append(f"{key}={float(metrics[key]):.3f}")
            if interesting:
                metric_suffix = " | " + ", ".join(interesting)
        if isinstance(loss, (int, float)):
            print(
                f"[{sample_slug}] iteration {iteration}/{total_iterations} "
                f"loss={float(loss):.3f}{metric_suffix}",
                flush=True,
            )
        else:
            print(f"[{sample_slug}] iteration {iteration}/{total_iterations}{metric_suffix}", flush=True)

    return callback


def _gpu_error_logger(sample_slug: str):
    def callback(exc: BaseException) -> None:
        print(
            f"[{sample_slug}] retryable GPU runtime error: "
            f"{exc.__class__.__name__}: {exc}",
            flush=True,
        )
        formatted = traceback.format_exception(type(exc), exc, exc.__traceback__)
        for line in "".join(formatted).rstrip().splitlines():
            print(f"[{sample_slug}] {line}", flush=True)

    return callback


def _teacher_overrides(args: argparse.Namespace) -> BackendSettingOverrides:
    return BackendSettingOverrides(
        defense_canvas=args.defense_canvas,
        force_full_strength=not args.allow_low_memory_fallback,
        immunization_iters=args.iters,
        eot_samples=args.eot_samples,
        max_prompt_variants=args.prompt_variants,
        denoiser_strength=args.denoiser_strength,
        reference_confusion_strength=args.reference_confusion,
        identity_drift_strength=args.identity_drift,
        semantic_boundary_strength=args.semantic_boundary,
        watermark_strength=args.watermark,
        tripwire_global_strength=args.global_anchor,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = (args.root_dir.expanduser().resolve() if args.root_dir else metadata_path.parent.resolve())
    bootstrap_feedback = Feedback(echo=True)
    pipeline = _get_pipeline()
    _report_pipeline_runtime_features(bootstrap_feedback)

    manifest_path = output_dir / "manifest.jsonl"
    failures_path = output_dir / "failures.jsonl"
    overrides = _teacher_overrides(args)
    rows = _load_metadata_rows(metadata_path)

    total_processed = 0
    total_succeeded = 0
    total_failed = 0

    for row_index, row in enumerate(rows, start=1):
        prompts = _expand_prompts(row)
        base_id = _row_id(row, row_index)
        image_path = _resolve_path(str(row["image"]), base_dir=dataset_root)
        mask_path = _resolve_path(str(row["mask"]), base_dir=dataset_root)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for row {row_index}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for row {row_index}: {mask_path}")

        image_bytes = image_path.read_bytes()
        row_mask_image = Image.open(mask_path).convert("L")
        mask_variants = _build_mask_variants(
            row_mask_image,
            int(row.get("mask_variant_count", args.mask_variant_count)),
            float(row.get("mask_variant_strength", args.mask_variant_strength)),
        )

        for mask_variant_name, mask_variant_image in mask_variants:
            mask_variant_bytes = _mask_to_png_bytes(mask_variant_image)
            for prompt_index, prompt in enumerate(prompts, start=1):
                if args.limit is not None and total_processed >= args.limit:
                    print(f"Reached --limit={args.limit}; stopping.", flush=True)
                    return 0

                total_processed += 1
                seed = int(row.get("seed", 1234))
                sample_slug = _sample_slug(base_id, prompt, prompt_index, mask_variant_name)
                sample_dir = samples_dir / sample_slug
                sample_dir.mkdir(parents=True, exist_ok=True)
                protected_path = sample_dir / "protected.png"

                if args.skip_existing and protected_path.exists():
                    print(f"[{sample_slug}] skipping existing sample.", flush=True)
                    continue

                print(f"[{sample_slug}] generating teacher sample...", flush=True)
                try:
                    sample_feedback = Feedback(echo=True)
                    prepared = _prepare_request_images(
                        image_bytes,
                        mask_variant_bytes,
                        working_resolution=str(row.get("working_resolution", args.working_resolution)),
                        fb=sample_feedback,
                    )
                    init_image = prepared.working_image
                    mask_image = prepared.working_edit_mask

                    effective_profile = _resolve_immunization_profile(
                        str(row.get("profile", args.profile)).strip().lower(),
                        sample_feedback,
                    )
                    preferred_resolution = _resolve_immunization_resolution(
                        effective_profile,
                        init_image.size,
                        str(row.get("defense_canvas", args.defense_canvas)),
                    )
                    _reset_cuda_peak_memory_stats()
                    immunization_image, immunization_mask = _resize_for_immunization_fallback(
                        init_image,
                        mask_image,
                        max_dimension=preferred_resolution,
                    )
                    immunization_config = _immunization_config_for_profile(
                        effective_profile,
                        immunization_image.size,
                    )
                    immunization_config = _apply_backend_setting_overrides(immunization_config, overrides)
                    immunization_config = replace(
                        immunization_config,
                        denoiser_early_timestep_bias=float(
                            row.get("early_step_bias", args.early_step_bias)
                        ),
                        mask_augmentation_strength=float(
                            row.get("mask_augment_strength", args.mask_augment_strength)
                        ),
                        mask_augmentation_count=int(
                            row.get("mask_augment_count", args.mask_augment_count)
                        ),
                        portrait_face_identity_strength=float(
                            row.get("portrait_face_identity", args.portrait_face_identity)
                        ),
                    )

                    progress_callback = _progress_logger(sample_slug)
                    gpu_error_callback = _gpu_error_logger(sample_slug)
                    immunized_working_image = _try_immunize_once(
                        immunization_image,
                        immunization_mask,
                        pipeline,
                        prompt=prompt,
                        guidance_scale=float(row.get("guidance_scale", args.guidance_scale)),
                        num_inference_steps=int(row.get("num_inference_steps", args.num_inference_steps)),
                        config=immunization_config,
                        seed=seed,
                        progress_callback=progress_callback,
                        error_callback=gpu_error_callback,
                    )
                    if immunized_working_image is None and not immunization_config.allow_low_memory_fallback:
                        raise RuntimeError("Full-strength immunization failed before fallback was allowed.")
                    if immunized_working_image is None:
                        fallback_config = _low_memory_immunization_config(immunization_config)
                        immunized_working_image = _try_immunize_once(
                            immunization_image,
                            immunization_mask,
                            pipeline,
                            prompt=prompt,
                            guidance_scale=float(row.get("guidance_scale", args.guidance_scale)),
                            num_inference_steps=int(row.get("num_inference_steps", args.num_inference_steps)),
                            config=fallback_config,
                            seed=seed,
                            progress_callback=progress_callback,
                            error_callback=gpu_error_callback,
                        )
                    if immunized_working_image is None:
                        raise RuntimeError("Immunization exhausted all fallback attempts.")
                    if immunized_working_image.size != init_image.size:
                        immunized_working_image = immunized_working_image.resize(
                            init_image.size,
                            resample=Image.LANCZOS,
                        )

                    protected_image = _finalize_output_image(
                        immunized_working_image,
                        prepared,
                        background=True,
                    )
                    protected_image.save(protected_path, format="PNG", optimize=True)
                    _save_delta(prepared.original_image, protected_image, sample_dir / "delta.npy")
                    _save_delta_visualization(
                        prepared.original_image,
                        protected_image,
                        sample_dir / "delta_vis.png",
                    )

                    metadata_payload = {
                        "sampleId": sample_slug,
                        "baseId": base_id,
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "maskVariant": mask_variant_name,
                        "prompt": prompt,
                        "promptFamily": _infer_prompt_family(prompt),
                        "seed": seed,
                        "profile": effective_profile,
                        "guidanceScale": float(row.get("guidance_scale", args.guidance_scale)),
                        "numInferenceSteps": int(row.get("num_inference_steps", args.num_inference_steps)),
                        "workingResolution": str(row.get("working_resolution", args.working_resolution)),
                        "teacherConfig": asdict(immunization_config),
                        "output": {
                            "protectedImage": str(protected_path),
                            "deltaNpy": str(sample_dir / "delta.npy"),
                            "deltaVisualization": str(sample_dir / "delta_vis.png"),
                        },
                    }
                    _write_json(sample_dir / "metadata.json", metadata_payload)
                    with manifest_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metadata_payload) + "\n")

                    _append_cuda_memory_feedback(sample_feedback, f"{sample_slug} teacher export")
                    total_succeeded += 1
                except Exception as exc:
                    total_failed += 1
                    failure_payload = {
                        "sampleId": sample_slug,
                        "baseId": base_id,
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "maskVariant": mask_variant_name,
                        "prompt": prompt,
                        "promptFamily": _infer_prompt_family(prompt),
                        "errorType": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    _write_json(sample_dir / "failure.json", failure_payload)
                    with failures_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure_payload) + "\n")
                    print(f"[{sample_slug}] FAILED: {exc}", flush=True)
                    if args.fail_fast:
                        raise

    print(
        f"Teacher export complete: {total_succeeded} succeeded, {total_failed} failed, "
        f"{total_processed} total prompt-expanded samples.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
