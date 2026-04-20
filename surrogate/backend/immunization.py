from dataclasses import dataclass, replace
import inspect
import math
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop
from vision_surrogates import (
    compute_face_id_embedding,
    compute_vision_surrogate_embedding,
    maybe_load_face_id_surrogate,
    maybe_load_vision_surrogate,
)

topil = ToPILImage()
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class ImmunizationConfig:
    profile_name: str = "stable_diffusion"
    eps: float = 0.12
    step_size: float = 0.01
    iters: int = 200
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    target_mode: str = "random"
    target_size: tuple[int, int] = (512, 512)
    target_strength: float = 1.0
    chaos_strength: float = 0.35
    denoiser_strength: float = 0.2
    denoiser_steps: int = 1
    denoiser_early_timestep_bias: float = 0.0
    eot_samples: int = 4
    resize_jitter: float = 0.15
    noise_strength: float = 0.03
    blur_kernel_size: int = 5
    mask_augmentation_strength: float = 0.0
    mask_augmentation_count: int = 0
    semantic_boundary_strength: float = 0.0
    semantic_ring_width: int = 9
    max_prompt_variants: int = 0
    compression_jitter_strength: float = 0.0
    subpixel_jitter: float = 0.0
    frequency_noise_strength: float = 0.0
    frequency_band_low: float = 0.08
    frequency_band_high: float = 0.28
    watermark_strength: float = 0.0
    region_priority_strength: float = 0.0
    priority_face_strength: float = 0.0
    priority_skin_strength: float = 0.0
    priority_text_strength: float = 0.0
    priority_logo_strength: float = 0.0
    texture_tracking_strength: float = 0.0
    edge_tracking_strength: float = 0.0
    reference_confusion_strength: float = 0.0
    identity_drift_strength: float = 0.0
    portrait_face_identity_strength: float = 0.0
    surrogate_face_id_strength: float = 0.0
    clip_vision_confusion_strength: float = 0.0
    dino_vision_confusion_strength: float = 0.0
    context_blend_strength: float = 0.0
    reference_region_count: int = 1
    tripwire_strength: float = 0.0
    tripwire_ring_strength: float = 0.0
    tripwire_anchor_strength: float = 0.0
    tripwire_anchor_count: int = 0
    tripwire_global_strength: float = 0.0
    tripwire_global_count: int = 0
    tripwire_tile_size: int = 32
    # Multi-scale descriptor losses (nano_banana_2+)
    multiscale_descriptor_strength: float = 0.0
    multiscale_descriptor_scales: tuple = (1.0, 0.5)
    # Background anchor descriptor confusion (nano_banana_2+)
    background_anchor_confusion_strength: float = 0.0
    background_anchor_count: int = 0
    # Multi-view cleanup proxies (nano_banana_2+)
    updown_scale_jitter: float = 0.0     # paired downscale/upscale simulation
    sharpen_proxy_strength: float = 0.0  # cheap unsharp-mask deartifact proxy
    teacher_purification_strength: float = 0.0
    teacher_purification_view_count: int = 0
    # Fallback scaling: when low-memory path fires, multiply profile-specific
    # losses by this factor instead of zeroing them.
    low_memory_loss_scale: float = 1.0
    # When False, any GPU-memory-triggered weakening is disabled and the run
    # fails closed so evaluation only reflects full-strength protection.
    allow_low_memory_fallback: bool = True


def dedupe_prompt_variants(candidates):
    prompt_variants = []
    seen = set()
    for candidate in candidates:
        normalized = " ".join((candidate or "").split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            prompt_variants.append(normalized)
    return prompt_variants


def is_sdxl_pipeline(pipeline):
    class_name = pipeline.__class__.__name__.lower()
    if "xl" in class_name:
        return True
    return getattr(pipeline, "text_encoder_2", None) is not None


def _detach_prompt_conditioning(prompt_conditioning):
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in prompt_conditioning.items()
    }


def _get_prompt_conditioning_dtype(prompt_conditioning):
    prompt_embeds = prompt_conditioning.get("prompt_embeds")
    if isinstance(prompt_embeds, torch.Tensor):
        return prompt_embeds.dtype
    pooled_prompt_embeds = prompt_conditioning.get("pooled_prompt_embeds")
    if isinstance(pooled_prompt_embeds, torch.Tensor):
        return pooled_prompt_embeds.dtype
    return torch.float32


def dilate_binary_mask(mask_tensor, radius):
    if radius <= 0:
        return mask_tensor
    kernel_size = int(radius) * 2 + 1
    return F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=radius).clamp(0, 1)


def erode_binary_mask(mask_tensor, radius):
    if radius <= 0:
        return mask_tensor
    kernel_size = int(radius) * 2 + 1
    return (1.0 - F.max_pool2d(1.0 - mask_tensor, kernel_size=kernel_size, stride=1, padding=radius)).clamp(0, 1)


def translate_binary_mask(mask_tensor, shift_x, shift_y):
    if shift_x == 0 and shift_y == 0:
        return mask_tensor
    translated = torch.zeros_like(mask_tensor)
    _, _, height, width = mask_tensor.shape

    src_x0 = max(0, -int(shift_x))
    src_x1 = min(width, width - int(shift_x)) if shift_x >= 0 else width
    dst_x0 = max(0, int(shift_x))
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    src_y0 = max(0, -int(shift_y))
    src_y1 = min(height, height - int(shift_y)) if shift_y >= 0 else height
    dst_y0 = max(0, int(shift_y))
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        translated[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = mask_tensor[:, :, src_y0:src_y1, src_x0:src_x1]
    return translated


def build_mask_augmentation_variants(mask_tensor, config):
    variants = [("original", mask_tensor.detach())]
    count = int(max(0, getattr(config, "mask_augmentation_count", 0)))
    strength = float(max(0.0, getattr(config, "mask_augmentation_strength", 0.0)))
    if count <= 0 or strength <= 0.0:
        return variants

    height, width = mask_tensor.shape[-2:]
    max_dim = max(height, width)
    radius = max(1, int(round(max_dim * min(0.03, 0.004 + 0.014 * strength))))
    shift = max(1, int(round(max_dim * min(0.015, 0.002 + 0.008 * strength))))

    candidates = [
        ("dilate", dilate_binary_mask(mask_tensor, radius)),
        ("erode", erode_binary_mask(mask_tensor, radius)),
        ("shift_right", translate_binary_mask(mask_tensor, shift, 0)),
        ("shift_left", translate_binary_mask(mask_tensor, -shift, 0)),
        ("shift_down", translate_binary_mask(mask_tensor, 0, shift)),
        ("shift_up", translate_binary_mask(mask_tensor, 0, -shift)),
    ]

    seen = {mask_tensor.detach().to(dtype=torch.uint8).cpu().numpy().tobytes()}
    for name, variant in candidates:
        normalized = (variant >= 0.5).to(dtype=torch.float32)
        key = normalized.to(dtype=torch.uint8).cpu().numpy().tobytes()
        if key in seen:
            continue
        seen.add(key)
        variants.append((name, normalized.detach()))
        if len(variants) >= count + 1:
            break
    return variants


def build_denoiser_timestep_weights(timesteps, early_bias):
    count = len(timesteps)
    if count <= 1 or early_bias <= 1e-6:
        return [1.0] * count
    ramp = torch.linspace(1.0, 0.35, steps=count, dtype=torch.float32)
    weights = ramp.pow(1.0 + float(early_bias))
    weights = weights / weights.mean().clamp_min(1e-6)
    return [float(weight.item()) for weight in weights]


def _build_sdxl_add_time_ids(pipeline, image_size, device, dtype, do_classifier_free_guidance):
    width, height = image_size
    spatial_size = (height, width)
    add_time_ids_fn = getattr(pipeline, "_get_add_time_ids", None)
    if add_time_ids_fn is None:
        add_time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            device=device,
            dtype=dtype,
        )
        return duplicate_for_cfg(add_time_ids, do_classifier_free_guidance)

    params = inspect.signature(add_time_ids_fn).parameters
    kwargs = {}
    if "original_size" in params:
        kwargs["original_size"] = spatial_size
    if "crops_coords_top_left" in params:
        kwargs["crops_coords_top_left"] = (0, 0)
    if "target_size" in params:
        kwargs["target_size"] = spatial_size
    if "negative_original_size" in params:
        kwargs["negative_original_size"] = spatial_size
    if "negative_crops_coords_top_left" in params:
        kwargs["negative_crops_coords_top_left"] = (0, 0)
    if "negative_target_size" in params:
        kwargs["negative_target_size"] = spatial_size
    if "aesthetic_score" in params:
        kwargs["aesthetic_score"] = 6.0
    if "negative_aesthetic_score" in params:
        kwargs["negative_aesthetic_score"] = 2.5
    if "dtype" in params:
        kwargs["dtype"] = dtype
    if "text_encoder_projection_dim" in params:
        projection_dim = getattr(getattr(getattr(pipeline, "text_encoder_2", None), "config", None), "projection_dim", None)
        if projection_dim is not None:
            kwargs["text_encoder_projection_dim"] = projection_dim

    add_time_ids = add_time_ids_fn(**kwargs)
    if isinstance(add_time_ids, tuple):
        if len(add_time_ids) == 2:
            positive_time_ids, negative_time_ids = add_time_ids
            add_time_ids = (
                torch.cat([negative_time_ids, positive_time_ids], dim=0)
                if do_classifier_free_guidance
                else positive_time_ids
            )
        else:
            raise RuntimeError("Unexpected SDXL add_time_ids payload.")
    elif do_classifier_free_guidance:
        add_time_ids = duplicate_for_cfg(add_time_ids, True)

    return add_time_ids.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Internal base: all PGD infrastructure lives here so StableDiffusion can
# inherit from LayeredDefense without a circular dependency.
# ---------------------------------------------------------------------------

class _PGDBaseProfile:
    """Internal base. Not exposed in IMMUNIZATION_PROFILES."""

    name = "_pgd_base"

    def build_prompt_variants(self, prompt):
        return [(prompt or "").strip()]

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        return transformed.new_tensor(0.0), {}

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        return adversarial

    def summarize_profile(self, config):
        return {"profile": self.name, "mode": "standard"}

    def prepare_reference_data(
        self,
        init_image,
        mask_tensor,
        masked_image,
        pipeline,
        prompt,
        guidance_scale,
        num_inference_steps,
        config,
        device,
        dtype,
        seed=None,
    ):
        target_image = build_target_image(
            init_image,
            mode=config.target_mode,
            size=config.target_size,
            seed=seed,
        )
        prompt_variants = self.build_prompt_variants(prompt)
        if config.max_prompt_variants > 0:
            prompt_variants = prompt_variants[: config.max_prompt_variants]
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds_variants = [
            encode_prompt_variant(pipeline, variant, device, do_classifier_free_guidance)
            for variant in prompt_variants
        ]

        scheduler = pipeline.scheduler.__class__.from_config(pipeline.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=device)

        original_image_tensor = preprocess(init_image).to(device=device, dtype=dtype)
        original_latents = encode_image_latents(
            pipeline.vae,
            original_image_tensor,
        )
        mask_variants = build_mask_augmentation_variants(mask_tensor, config)
        denoiser_mask_views = []
        for name, variant_mask in mask_variants:
            variant_mask = variant_mask.to(device=device, dtype=torch.float32)
            variant_masked_image = original_image_tensor * (variant_mask < 0.5)
            variant_masked_latents = encode_image_latents(pipeline.vae, variant_masked_image)
            denoiser_mask_views.append(
                {
                    "name": name,
                    "mask_latents": prepare_mask_latents(variant_mask, dtype, device, do_classifier_free_guidance).detach(),
                    "masked_image_latents": duplicate_for_cfg(
                        variant_masked_latents,
                        do_classifier_free_guidance,
                    ).detach(),
                }
            )
        mask_latents = denoiser_mask_views[0]["mask_latents"]
        add_time_ids = None
        if is_sdxl_pipeline(pipeline) and prompt_embeds_variants:
            add_time_ids = _build_sdxl_add_time_ids(
                pipeline,
                init_image.size,
                device,
                _get_prompt_conditioning_dtype(prompt_embeds_variants[0]),
                do_classifier_free_guidance,
            )

        denoiser_timesteps = select_proxy_timesteps(scheduler.timesteps, config.denoiser_steps)
        denoiser_timestep_weights = build_denoiser_timestep_weights(
            denoiser_timesteps,
            getattr(config, "denoiser_early_timestep_bias", 0.0),
        )
        original_noise_predictions = []
        noises = []
        with torch.no_grad():
            for step_index, timestep in enumerate(denoiser_timesteps):
                generator = torch.Generator(device=device)
                generator.manual_seed((seed or 0) + step_index)
                noise = torch.randn(original_latents.shape, generator=generator, device=device, dtype=dtype)
                noises.append(noise)
                step_predictions = []
                for prompt_embeds in prompt_embeds_variants:
                    prompt_predictions = []
                    for mask_view in denoiser_mask_views:
                        prompt_predictions.append(
                            predict_guided_noise(
                                pipeline,
                                scheduler,
                                original_latents,
                                mask_view["mask_latents"],
                                mask_view["masked_image_latents"],
                                prompt_embeds,
                                add_time_ids,
                                timestep,
                                guidance_scale,
                                do_classifier_free_guidance,
                                noise,
                            ).detach()
                        )
                    step_predictions.append(prompt_predictions)
                original_noise_predictions.append(step_predictions)

        reference_data = {
            "target_image": target_image,
            "targets": encode_target(pipeline.vae, target_image, device, dtype),
            "original_latents": original_latents.detach(),
            "masked_image": masked_image.detach(),
            "mask_tensor": mask_tensor.detach(),
            "mask_latents": mask_latents.detach(),
            "denoiser_mask_views": denoiser_mask_views,
            "prompt_embeds_variants": [_detach_prompt_conditioning(pe) for pe in prompt_embeds_variants],
            "guidance_scale": guidance_scale,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "scheduler": scheduler,
            "denoiser_timesteps": denoiser_timesteps,
            "denoiser_timestep_weights": denoiser_timestep_weights,
            "denoiser_noises": noises,
            "original_noise_predictions": original_noise_predictions,
        }
        if add_time_ids is not None:
            reference_data["add_time_ids"] = add_time_ids.detach()
        return self.augment_reference_data(
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
            seed=seed,
        )

    def compute_loss(self, transformed, reference_data, config, pipeline, dtype):
        transformed_latents = encode_image_latents(pipeline.vae, transformed)
        target_loss = (transformed_latents - reference_data["targets"]).norm()
        chaos_loss = (transformed_latents - reference_data["original_latents"]).norm()
        denoiser_loss = transformed_latents.new_tensor(0.0)
        if config.denoiser_strength > 0 and reference_data["denoiser_timesteps"]:
            transformed_masked_latents = duplicate_for_cfg(
                transformed_latents,
                reference_data["do_classifier_free_guidance"],
            )
            denoiser_total = transformed_latents.new_tensor(0.0)
            denoiser_weight_total = 0.0
            denoiser_mask_views = reference_data.get("denoiser_mask_views") or []
            timestep_weights = reference_data.get("denoiser_timestep_weights") or [1.0] * len(reference_data["denoiser_timesteps"])
            for timestep, timestep_weight, noise, original_noise_predictions in zip(
                reference_data["denoiser_timesteps"],
                timestep_weights,
                reference_data["denoiser_noises"],
                reference_data["original_noise_predictions"],
            ):
                for prompt_embeds, original_noise_predictions_for_prompt in zip(
                    reference_data["prompt_embeds_variants"],
                    original_noise_predictions,
                ):
                    for mask_view, original_noise_prediction in zip(
                        denoiser_mask_views,
                        original_noise_predictions_for_prompt,
                    ):
                        guided_noise = predict_guided_noise(
                            pipeline,
                            reference_data["scheduler"],
                            transformed_latents,
                            mask_view["mask_latents"],
                            mask_view["masked_image_latents"],
                            prompt_embeds,
                            reference_data.get("add_time_ids"),
                            timestep,
                            reference_data["guidance_scale"],
                            reference_data["do_classifier_free_guidance"],
                            noise,
                        )
                        denoiser_total = denoiser_total + float(timestep_weight) * (
                            guided_noise - original_noise_prediction
                        ).norm()
                        denoiser_weight_total += float(timestep_weight)
            if denoiser_weight_total > 0:
                denoiser_loss = denoiser_total / float(denoiser_weight_total)

        extra_loss, extra_metrics = self.compute_extra_loss(
            transformed,
            reference_data,
            config,
            pipeline,
            dtype,
        )
        total_loss = (
            config.target_strength * target_loss
            - config.chaos_strength * chaos_loss
            - config.denoiser_strength * denoiser_loss
            - extra_loss
        )
        metrics = {
            "target": target_loss.detach(),
            "chaos": chaos_loss.detach(),
            "denoiser": denoiser_loss.detach(),
            **extra_metrics,
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Layered defense base: adds boundary/priority/watermark losses on top of PGD.
# ---------------------------------------------------------------------------

class LayeredDefenseImmunizationProfile(_PGDBaseProfile):
    name = "layered_defense_base"
    default_prompt = "Edit the protected subject while preserving the rest of the image."
    prompt_suffixes = ()

    def build_prompt_variants(self, prompt):
        base_prompt = (prompt or "").strip() or self.default_prompt
        candidates = [base_prompt]
        candidates.extend(suffix.format(prompt=base_prompt) for suffix in self.prompt_suffixes)
        return dedupe_prompt_variants(candidates)

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        boundary_ring = build_boundary_ring(mask_tensor.detach(), config.semantic_ring_width).detach()
        protected_mask = (1 - mask_tensor).detach()
        reference_data["boundary_ring"] = boundary_ring
        reference_data["protected_mask"] = protected_mask
        reference_data["original_masked_image"] = masked_image.detach()

        if (
            config.region_priority_strength > 0
            or config.priority_face_strength > 0
            or config.priority_skin_strength > 0
            or config.priority_text_strength > 0
            or config.priority_logo_strength > 0
            or config.portrait_face_identity_strength > 0
        ):
            reference_data.update(
                build_region_priority_maps(
                    init_image,
                    protected_mask,
                    boundary_ring,
                    config,
                    device,
                )
            )

        if config.watermark_strength > 0:
            watermark_band_mask = build_frequency_band_mask(
                protected_mask.shape[-2],
                protected_mask.shape[-1],
                config.frequency_band_low,
                config.frequency_band_high,
                device,
            )
            reference_data["watermark_band_mask"] = watermark_band_mask.detach()
            reference_data["watermark_template"] = build_watermark_template(
                protected_mask.shape[-2],
                protected_mask.shape[-1],
                seed,
                config.frequency_band_low,
                config.frequency_band_high,
                device,
            ).detach()

        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss = transformed.new_tensor(0.0)
        metrics = {}
        original_masked_image = reference_data.get("original_masked_image")
        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")

        if (
            config.semantic_boundary_strength > 0
            and original_masked_image is not None
            and protected_mask is not None
            and boundary_ring is not None
        ):
            boundary_change = ((transformed - original_masked_image).abs() * boundary_ring).mean()
            protected_change = ((transformed - original_masked_image).abs() * protected_mask).mean()
            extra_loss = extra_loss + config.semantic_boundary_strength * (boundary_change + 0.25 * protected_change)
            metrics["semantic"] = boundary_change.detach()
            metrics["protected"] = protected_change.detach()

        priority_map = reference_data.get("region_priority_map")
        if config.region_priority_strength > 0 and priority_map is not None and original_masked_image is not None:
            priority_change = ((transformed - original_masked_image).abs() * priority_map).mean()
            extra_loss = extra_loss + config.region_priority_strength * priority_change
            metrics["priority"] = priority_change.detach()

        watermark_band_mask = reference_data.get("watermark_band_mask")
        watermark_template = reference_data.get("watermark_template")
        if config.watermark_strength > 0 and watermark_band_mask is not None and watermark_template is not None:
            watermark_mask = priority_map if priority_map is not None else protected_mask
            watermark_alignment = compute_watermark_alignment(
                transformed,
                watermark_band_mask,
                watermark_template,
                watermark_mask,
            )
            extra_loss = extra_loss + config.watermark_strength * watermark_alignment
            metrics["watermark"] = watermark_alignment.detach()

        return extra_loss, metrics


# ---------------------------------------------------------------------------
# stable_diffusion profile — upgraded.
# Now inherits LayeredDefense and adds:
#   • prompt variants covering DeeVid's documented attack surface
#   • mild global background anchor tripwire (deters outpainting / full-redraw)
# ---------------------------------------------------------------------------

class StableDiffusionImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "stable_diffusion"
    default_prompt = "Edit the image realistically."
    # Covers: layout-preserving restyle, subject replacement, object removal,
    # background swap, reference-guided edit (all documented DeeVid modes).
    prompt_suffixes = (
        "{prompt} Restyle the full image while keeping composition and subject placement identical.",
        "{prompt} Replace the protected subject with a different plausible subject while keeping lighting and background consistent.",
        "{prompt} Remove the protected subject and fill the region naturally while keeping the rest unchanged.",
        "{prompt} Change only the background while keeping the protected subject identical.",
        "{prompt} Use the provided reference images to guide the edit while preserving the protected subject.",
    )

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        # Inherit boundary/priority/watermark setup from LayeredDefense.
        reference_data = super().augment_reference_data(
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
            seed=seed,
        )

        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")

        # Mild global background anchor: 1-2 sparse tiles in the unprotected
        # region.  Provides a weak signal against full-image redraws / outpainting
        # without meaningfully raising visible artifact levels.
        if (
            config.tripwire_global_strength > 0
            and config.tripwire_global_count > 0
            and protected_mask is not None
        ):
            # Keep "global" anchors inside perturbable pixels. Anchors placed
            # entirely outside the protected region would be constant after the
            # final mask blend and contribute no useful gradient.
            bg_candidate = protected_mask.to(device=device, dtype=torch.float32)
            if boundary_ring is not None and float(boundary_ring.mean().item()) > 1e-5:
                edge_candidate = (boundary_ring.to(device=device, dtype=torch.float32) * bg_candidate).clamp(0, 1)
                if float(edge_candidate.mean().item()) > 1e-5:
                    bg_candidate = normalize_map(edge_candidate)
            global_anchor_mask = build_sparse_anchor_mask(
                bg_candidate.to(device=device, dtype=torch.float32),
                seed=(seed or 0) + 4421,
                tile_size=config.tripwire_tile_size,
                anchor_count=config.tripwire_global_count,
            )
            if float(global_anchor_mask.mean().item()) > 0.0:
                reference_data["sd_global_anchor_mask"] = global_anchor_mask.detach()
                reference_data["sd_global_anchor_template"] = build_tripwire_template(
                    protected_mask.shape[-2],
                    protected_mask.shape[-1],
                    seed=(seed or 0) + 9973,
                    device=device,
                    phase_offset=4,
                ).detach()

        original_image_tensor = None
        if (
            (config.reference_confusion_strength > 0 or config.portrait_face_identity_strength > 0)
            and protected_mask is not None
        ):
            original_image_tensor = preprocess(init_image).to(device=device, dtype=torch.float32)
            reference_data["sd_original_image_tensor"] = original_image_tensor.detach()
        # Light descriptor confusion on the protected region.
        if config.reference_confusion_strength > 0 and protected_mask is not None and original_image_tensor is not None:
            reference_data["sd_protected_descriptor"] = compute_masked_descriptor(
                original_image_tensor,
                protected_mask.to(device=device, dtype=torch.float32),
            ).detach()
            if boundary_ring is not None and float(boundary_ring.mean().item()) > 1e-5:
                reference_data["sd_boundary_descriptor"] = compute_masked_descriptor(
                    original_image_tensor,
                    boundary_ring.to(device=device, dtype=torch.float32),
                ).detach()

        if config.portrait_face_identity_strength > 0:
            face_proxy_map = reference_data.get("face_proxy_map")
            if (
                face_proxy_map is not None
                and original_image_tensor is not None
                and float(face_proxy_map.mean().item()) > 1e-5
            ):
                face_mask = normalize_map(
                    face_proxy_map.to(device=device, dtype=torch.float32)
                    * protected_mask.to(device=device, dtype=torch.float32)
                )
                if float(face_mask.mean().item()) > 1e-5:
                    reference_data["sd_face_proxy_mask"] = face_mask.detach()
                    reference_data["sd_face_descriptor"] = compute_masked_descriptor(
                        original_image_tensor,
                        face_mask,
                    ).detach()

        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss, metrics = super().compute_extra_loss(
            transformed, reference_data, config, pipeline, dtype,
        )

        # Light descriptor confusion on protected region + boundary.
        if config.reference_confusion_strength > 0:
            image_f32 = transformed.to(dtype=torch.float32)
            protected_mask = reference_data.get("protected_mask")
            ref_desc = reference_data.get("sd_protected_descriptor")
            if protected_mask is not None and ref_desc is not None:
                cur_desc = compute_masked_descriptor(image_f32, protected_mask)
                confusion = descriptor_distance(cur_desc, ref_desc)
                extra_loss = extra_loss + config.reference_confusion_strength * confusion
                metrics["sd_confusion"] = confusion.detach()

            boundary_ring = reference_data.get("boundary_ring")
            bnd_desc = reference_data.get("sd_boundary_descriptor")
            if boundary_ring is not None and bnd_desc is not None:
                cur_bnd = compute_masked_descriptor(image_f32, boundary_ring)
                bnd_confusion = descriptor_distance(cur_bnd, bnd_desc)
                # Half weight — boundary is a secondary signal.
                extra_loss = extra_loss + 0.5 * config.reference_confusion_strength * bnd_confusion
                metrics["sd_bnd_confusion"] = bnd_confusion.detach()

        # Light identity drift under subpixel/compression transforms.
        if config.identity_drift_strength > 0:
            image_f32 = transformed.to(dtype=torch.float32)
            protected_mask = reference_data.get("protected_mask")
            ref_desc = reference_data.get("sd_protected_descriptor")
            if protected_mask is not None and ref_desc is not None:
                drift_view = build_identity_drift_view(image_f32, config)
                drift_desc = compute_masked_descriptor(drift_view, protected_mask)
                drift_loss = descriptor_distance(drift_desc, ref_desc)
                extra_loss = extra_loss + config.identity_drift_strength * drift_loss
                metrics["sd_drift"] = drift_loss.detach()

        if config.portrait_face_identity_strength > 0:
            image_f32 = transformed.to(dtype=torch.float32)
            face_mask = reference_data.get("sd_face_proxy_mask")
            face_ref = reference_data.get("sd_face_descriptor")
            if face_mask is not None and face_ref is not None:
                face_desc = compute_masked_descriptor(image_f32, face_mask)
                face_confusion = descriptor_distance(face_desc, face_ref)
                extra_loss = extra_loss + config.portrait_face_identity_strength * face_confusion
                metrics["sd_face"] = face_confusion.detach()

                face_drift_view = build_identity_drift_view(image_f32, config)
                face_drift_desc = compute_masked_descriptor(face_drift_view, face_mask)
                face_drift = descriptor_distance(face_drift_desc, face_ref)
                extra_loss = extra_loss + 0.5 * config.portrait_face_identity_strength * face_drift
                metrics["sd_face_drift"] = face_drift.detach()

        # Mild global background anchor against full-image redraw / outpainting.
        global_mask = reference_data.get("sd_global_anchor_mask")
        global_template = reference_data.get("sd_global_anchor_template")
        if config.tripwire_global_strength > 0 and global_mask is not None and global_template is not None:
            global_align = compute_tripwire_alignment(
                rgb_to_luma(transformed.to(dtype=torch.float32)),
                global_template,
                global_mask,
            )
            extra_loss = extra_loss + config.tripwire_global_strength * global_align
            metrics["sd_global"] = global_align.detach()

        return extra_loss, metrics

    def summarize_profile(self, config):
        return {
            "profile": self.name,
            "mode": "standard_layered",
            "globalAnchorStrength": float(config.tripwire_global_strength),
        }


# ---------------------------------------------------------------------------
# Scaffold profiles — unchanged except they now inherit from the right base.
# ---------------------------------------------------------------------------

class FullRegenerationScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "full_regeneration_scaffold"
    default_prompt = "Regenerate the whole photo while preserving the same scene."
    prompt_suffixes = (
        "{prompt} Re-render the full image while keeping the same people and objects recognizable.",
        "{prompt} Keep the scene layout recognizable, but resample the overall image appearance.",
    )


class InstructionEditingScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "instruction_editing_scaffold"
    default_prompt = "Follow an edit instruction while preserving a realistic photograph."
    prompt_suffixes = (
        "{prompt} Change only the requested attribute and keep all other entities consistent.",
        "{prompt} Add or remove the requested object while preserving the realism of the rest of the photo.",
    )


class ControlNetScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "controlnet_scaffold"
    default_prompt = "Edit the appearance of the protected subject while keeping pose and geometry fixed."
    prompt_suffixes = (
        "{prompt} Keep the same pose, depth, and edges, but alter texture and identity cues.",
        "{prompt} Preserve structure and composition while changing only appearance details.",
    )

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss, metrics = super().compute_extra_loss(
            transformed, reference_data, config, pipeline, dtype,
        )
        original_masked_image = reference_data.get("original_masked_image")
        protected_mask = reference_data.get("protected_mask")
        if (
            config.texture_tracking_strength > 0
            and original_masked_image is not None
            and protected_mask is not None
        ):
            texture_change = compute_high_frequency_change(
                transformed, original_masked_image, protected_mask,
            )
            extra_loss = extra_loss + config.texture_tracking_strength * texture_change
            metrics["texture"] = texture_change.detach()
        return extra_loss, metrics


class StyleTransferScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "style_transfer_scaffold"
    default_prompt = "Restyle the image while preserving composition."
    prompt_suffixes = (
        "{prompt} Change the color palette and texture style while keeping the same content.",
        "{prompt} Transform the aesthetic style but preserve the same scene edges and object placement.",
    )

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss, metrics = super().compute_extra_loss(
            transformed, reference_data, config, pipeline, dtype,
        )
        original_masked_image = reference_data.get("original_masked_image")
        protected_mask = reference_data.get("protected_mask")
        if config.edge_tracking_strength > 0 and original_masked_image is not None and protected_mask is not None:
            edge_change = compute_edge_change(transformed, original_masked_image, protected_mask)
            extra_loss = extra_loss + config.edge_tracking_strength * edge_change
            metrics["edges"] = edge_change.detach()
        return extra_loss, metrics


class TextAwareScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "text_aware_scaffold"
    default_prompt = "Edit the image while keeping text, logos, and overlays consistent."
    prompt_suffixes = (
        "{prompt} Rewrite or remove nearby text while preserving the surrounding image realism.",
        "{prompt} Modify the overlay or logo region while keeping the rest of the image unchanged.",
    )


class AdversarialHardenedScaffoldImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "adversarial_hardened_scaffold"
    default_prompt = "Edit the image with a tuned or adversarially trained pipeline."
    prompt_suffixes = (
        "{prompt} Assume the editor was fine-tuned to preserve watermark-like artifacts while changing content.",
        "{prompt} Apply a highly targeted edit that tries to preserve visible quality and bypass defenses.",
        "{prompt} Keep the scene realistic while performing a targeted identity or object manipulation.",
    )


# ---------------------------------------------------------------------------
# NanoBananaExperimental — unchanged.
# ---------------------------------------------------------------------------

class NanoBananaExperimentalImmunizationProfile(LayeredDefenseImmunizationProfile):
    name = "nano_banana_experimental"
    default_prompt = "Edit the protected subject while preserving the rest of the image."
    prompt_suffixes = (
        "{prompt} Change only the protected subject and preserve the rest of the photo.",
        "{prompt} Replace the protected subject with a different plausible subject while keeping lighting and background consistent.",
        "{prompt} Remove the protected subject and fill the region realistically while preserving the rest of the image.",
        "{prompt} Rewrite nearby text or logos only where needed while keeping the rest of the scene photorealistic.",
    )

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        delta = (adversarial - base_image) * perturbation_mask
        if config.blur_kernel_size > 1:
            delta = F.avg_pool2d(
                delta,
                kernel_size=config.blur_kernel_size,
                stride=1,
                padding=config.blur_kernel_size // 2,
            )
        delta = delta.clamp(min=-config.eps * 0.55, max=config.eps * 0.55)
        smoothed = base_image + (0.7 * delta)
        return smoothed.clamp(min=config.clamp_min, max=config.clamp_max)


# ---------------------------------------------------------------------------
# nano_banana_2 — materially upgraded.
#
# New capabilities vs the old regular profile:
#   1. Expanded prompt variants covering OpenArt's documented attack surface.
#   2. Multi-scale descriptor losses on protected, boundary, and sparse
#      background anchors — confuses reference-image and character-consistency
#      editors that embed descriptors at multiple spatial resolutions.
#   3. Weak always-on protected + ring tripwires (backported from hard_block at
#      ~40 % of hard_block strength).
#   4. Multi-view cleanup proxies: paired upscale/downscale and optional cheap
#      unsharp-mask sharpen pass, simulating OpenArt-style pipeline cleanup.
#   5. Fallback: low-memory path scales losses by low_memory_loss_scale instead
#      of zeroing them.
# ---------------------------------------------------------------------------

class NanoBanana2ImmunizationProfile(NanoBananaExperimentalImmunizationProfile):
    name = "nano_banana_2"
    default_prompt = "Edit the protected subject while preserving the rest of the image."
    # Expanded to cover OpenArt's documented modes: chat edit, character
    # consistency, face fix, text/logo cleanup, background swap, object removal.
    prompt_suffixes = (
        "{prompt} Change only the protected subject and preserve the rest of the photo.",
        "{prompt} Replace the protected subject with a different plausible subject while keeping lighting and background consistent.",
        "{prompt} Remove the protected subject and fill the region realistically while preserving the rest of the image.",
        "{prompt} Rewrite nearby text or logos only where needed while keeping the rest of the scene photorealistic.",
        "{prompt} Use multiple reference images to preserve character consistency while editing only the protected subject.",
        "{prompt} Fix or enhance the face of the protected subject while keeping all surrounding context identical.",
        "{prompt} Clean up or replace text and logo overlays while leaving the protected subject and background unchanged.",
        "{prompt} Swap only the background while keeping the protected subject pixel-accurate.",
        "{prompt} Remove the specified object from the scene and fill the gap plausibly while keeping the rest untouched.",
    )

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        # Call LayeredDefense → boundary/priority/watermark.
        reference_data = LayeredDefenseImmunizationProfile.augment_reference_data(
            self,
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
            seed=seed,
        )

        original_image_tensor = preprocess(init_image).to(device=device, dtype=torch.float32)
        reference_data["original_image_tensor"] = original_image_tensor.detach()

        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")

        # --- Per-region descriptors (existing logic) ---
        reference_region_masks = build_reference_region_masks(reference_data, config)
        reference_data["reference_region_masks"] = [r.detach() for r in reference_region_masks]
        reference_data["reference_region_descriptors"] = [
            compute_masked_descriptor(original_image_tensor, r).detach()
            for r in reference_region_masks
        ]

        # --- Multi-scale descriptors for protected region ---
        # Encoders in reference-guided editors typically embed at multiple
        # spatial scales; mismatching across scales confuses all of them.
        if config.multiscale_descriptor_strength > 0 and protected_mask is not None:
            scales = config.multiscale_descriptor_scales or (1.0, 0.5)
            reference_data["ms_protected_descriptor"] = compute_multiscale_descriptor(
                original_image_tensor,
                protected_mask.to(device=device, dtype=torch.float32),
                scales=scales,
            ).detach()

        # --- Background anchor descriptor confusion ---
        # Sparse tiles sampled from the unprotected region give reference-image
        # editors conflicting context signals for the surrounding scene,
        # disrupting outpaint / background-swap coherence.
        if config.background_anchor_confusion_strength > 0 and config.background_anchor_count > 0:
            # Use sparse context anchors inside the perturbable region so this
            # term has gradient. We bias toward the protected-side boundary
            # band because reference-guided editors read those context cues.
            bg_candidate = protected_mask.to(device=device, dtype=torch.float32) if protected_mask is not None else torch.zeros_like(mask_tensor)
            if boundary_ring is not None and float(boundary_ring.mean().item()) > 1e-5:
                edge_candidate = (bg_candidate * boundary_ring.to(device=device, dtype=torch.float32)).clamp(0, 1)
                if float(edge_candidate.mean().item()) > 1e-5:
                    bg_candidate = normalize_map(edge_candidate)
            bg_anchor_mask = build_sparse_anchor_mask(
                bg_candidate.to(device=device, dtype=torch.float32),
                seed=(seed or 0) + 3571,
                tile_size=config.tripwire_tile_size,
                anchor_count=config.background_anchor_count,
            )
            if float(bg_anchor_mask.mean().item()) > 0.0:
                reference_data["bg_anchor_mask"] = bg_anchor_mask.detach()
                reference_data["bg_anchor_descriptor"] = compute_masked_descriptor(
                    original_image_tensor,
                    bg_anchor_mask.to(device=device, dtype=torch.float32),
                ).detach()

        # --- Boundary descriptor for context blending ---
        if (
            config.context_blend_strength > 0
            and boundary_ring is not None
            and float(boundary_ring.mean().item()) > 1e-5
        ):
            reference_data["boundary_descriptor"] = compute_masked_descriptor(
                original_image_tensor,
                boundary_ring,
            ).detach()

        # --- Optional vision/reference encoder surrogates ---
        face_proxy_map = reference_data.get("face_proxy_map")
        face_mask = None
        if face_proxy_map is not None and protected_mask is not None:
            face_mask = normalize_map(
                face_proxy_map.to(device=device, dtype=torch.float32)
                * protected_mask.to(device=device, dtype=torch.float32)
            )
            if float(face_mask.mean().item()) > 1e-5:
                reference_data["surrogate_face_mask"] = face_mask.detach()
            else:
                face_mask = None

        clip_surrogate = None
        dino_surrogate = None
        face_id_surrogate = None
        if config.clip_vision_confusion_strength > 0:
            clip_surrogate = maybe_load_vision_surrogate("clip", torch.device(device))
            if clip_surrogate is not None:
                reference_data["clip_surrogate"] = clip_surrogate
                clip_descriptor = compute_vision_surrogate_embedding(
                    clip_surrogate,
                    original_image_tensor,
                    protected_mask,
                )
                if clip_descriptor is not None:
                    reference_data["clip_protected_descriptor"] = clip_descriptor.detach()

        if config.dino_vision_confusion_strength > 0:
            dino_surrogate = maybe_load_vision_surrogate("dino", torch.device(device))
            if dino_surrogate is not None:
                reference_data["dino_surrogate"] = dino_surrogate
                dino_descriptor = compute_vision_surrogate_embedding(
                    dino_surrogate,
                    original_image_tensor,
                    protected_mask,
                )
                if dino_descriptor is not None:
                    reference_data["dino_protected_descriptor"] = dino_descriptor.detach()

        if config.surrogate_face_id_strength > 0 and face_mask is not None:
            face_id_surrogate = maybe_load_face_id_surrogate(torch.device(device))
            if face_id_surrogate is not None:
                reference_data["face_id_surrogate"] = face_id_surrogate
                face_id_descriptor = compute_face_id_embedding(
                    face_id_surrogate,
                    original_image_tensor,
                    face_mask,
                )
                if face_id_descriptor is not None:
                    reference_data["surrogate_face_id_descriptor"] = face_id_descriptor.detach()

        # --- Weak always-on tripwires (backported from hard_block at ~40%) ---
        # These fire unconditionally in the regular profile so the optimizer
        # cannot find a low-effort bypass via the fallback path.
        if (
            (config.tripwire_strength > 0 or config.tripwire_ring_strength > 0)
            and protected_mask is not None
        ):
            h, w = protected_mask.shape[-2:]
            reference_data["tripwire_protected_template"] = build_tripwire_template(
                h, w, seed=seed, device=device, phase_offset=0,
            ).detach()
            if config.tripwire_ring_strength > 0 and boundary_ring is not None:
                ring_mask = normalize_map(
                    boundary_ring.to(device=device, dtype=torch.float32)
                    * protected_mask.to(device=device, dtype=torch.float32)
                )
                reference_data["tripwire_ring_mask"] = ring_mask.detach()
                reference_data["tripwire_ring_template"] = build_tripwire_template(
                    h, w, seed=(seed or 0) + 101, device=device, phase_offset=1,
                ).detach()

        if config.tripwire_anchor_strength > 0 and config.tripwire_anchor_count > 0 and protected_mask is not None:
            h, w = protected_mask.shape[-2:]
            anchor_mask = build_sparse_anchor_mask(
                protected_mask.to(device=device, dtype=torch.float32),
                seed=seed,
                tile_size=config.tripwire_tile_size,
                anchor_count=config.tripwire_anchor_count,
            )
            reference_data["tripwire_anchor_mask"] = anchor_mask.detach()
            reference_data["tripwire_anchor_template"] = build_tripwire_template(
                h, w, seed=(seed or 0) + 211, device=device, phase_offset=2,
            ).detach()

        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        # LayeredDefense handles boundary/priority/watermark.
        extra_loss, metrics = LayeredDefenseImmunizationProfile.compute_extra_loss(
            self, transformed, reference_data, config, pipeline, dtype,
        )

        image_tensor = transformed.to(dtype=torch.float32)

        # --- Multi-view cleanup proxies ---
        # Simulate OpenArt-style upscale/downscale and deartifact passes so
        # the perturbation survives those pipelines better.
        if config.updown_scale_jitter > 0:
            image_tensor = apply_updown_scale_proxy(image_tensor, scale=1.0 - config.updown_scale_jitter)
        if config.sharpen_proxy_strength > 0:
            image_tensor = apply_sharpen_proxy(image_tensor, strength=config.sharpen_proxy_strength)

        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")
        reference_region_masks = reference_data.get("reference_region_masks") or []
        reference_region_descriptors = reference_data.get("reference_region_descriptors") or []

        primary_mask = None
        primary_descriptor = None

        # --- Per-region descriptor confusion ---
        if (
            config.reference_confusion_strength > 0
            and reference_region_masks
            and reference_region_descriptors
        ):
            confusion_losses = []
            for region_mask, reference_descriptor in zip(
                reference_region_masks, reference_region_descriptors,
            ):
                transformed_descriptor = compute_masked_descriptor(image_tensor, region_mask)
                confusion_losses.append(descriptor_distance(transformed_descriptor, reference_descriptor))
                if primary_mask is None:
                    primary_mask = region_mask
                    primary_descriptor = transformed_descriptor

            reference_confusion = torch.stack(confusion_losses).mean()
            extra_loss = extra_loss + config.reference_confusion_strength * reference_confusion
            metrics["reference"] = reference_confusion.detach()

        if primary_mask is None:
            primary_mask = protected_mask
        if primary_descriptor is None and primary_mask is not None:
            primary_descriptor = compute_masked_descriptor(image_tensor, primary_mask)

        # --- Multi-scale descriptor loss ---
        # Pushes the defended image's multi-scale signature away from the
        # original; reference-image editors that encode at multiple resolutions
        # all receive conflicting signals simultaneously.
        ms_ref = reference_data.get("ms_protected_descriptor")
        if config.multiscale_descriptor_strength > 0 and ms_ref is not None and protected_mask is not None:
            scales = config.multiscale_descriptor_scales or (1.0, 0.5)
            ms_cur = compute_multiscale_descriptor(image_tensor, protected_mask, scales=scales)
            ms_loss = descriptor_distance(ms_cur, ms_ref.to(device=image_tensor.device, dtype=image_tensor.dtype))
            extra_loss = extra_loss + config.multiscale_descriptor_strength * ms_loss
            metrics["ms_desc"] = ms_loss.detach()

        # --- Background anchor descriptor confusion ---
        # Corrupt the descriptor fingerprint of sparse background tiles so
        # scene-context readers in reference-guided editors disagree with what
        # they see in the undefended background.
        bg_anchor_mask = reference_data.get("bg_anchor_mask")
        bg_anchor_ref = reference_data.get("bg_anchor_descriptor")
        if (
            config.background_anchor_confusion_strength > 0
            and bg_anchor_mask is not None
            and bg_anchor_ref is not None
        ):
            bg_cur = compute_masked_descriptor(image_tensor, bg_anchor_mask)
            bg_loss = descriptor_distance(bg_cur, bg_anchor_ref.to(device=image_tensor.device, dtype=image_tensor.dtype))
            extra_loss = extra_loss + config.background_anchor_confusion_strength * bg_loss
            metrics["bg_anchor"] = bg_loss.detach()

        # --- Identity drift under transforms ---
        if config.identity_drift_strength > 0 and primary_mask is not None and primary_descriptor is not None:
            identity_view = build_identity_drift_view(image_tensor, config)
            identity_descriptor = compute_masked_descriptor(identity_view, primary_mask)
            identity_drift = descriptor_distance(identity_descriptor, primary_descriptor)
            extra_loss = extra_loss + config.identity_drift_strength * identity_drift
            metrics["identity"] = identity_drift.detach()

        # --- Context blend ---
        boundary_descriptor = reference_data.get("boundary_descriptor")
        if (
            config.context_blend_strength > 0
            and boundary_descriptor is not None
            and primary_descriptor is not None
        ):
            context_similarity = descriptor_similarity(primary_descriptor, boundary_descriptor)
            extra_loss = extra_loss + config.context_blend_strength * context_similarity
            metrics["context"] = context_similarity.detach()

        # --- Vision encoder surrogates ---
        clip_surrogate = reference_data.get("clip_surrogate")
        clip_ref = reference_data.get("clip_protected_descriptor")
        if (
            config.clip_vision_confusion_strength > 0
            and clip_surrogate is not None
            and clip_ref is not None
            and protected_mask is not None
        ):
            clip_cur = compute_vision_surrogate_embedding(clip_surrogate, image_tensor, protected_mask)
            if clip_cur is not None:
                clip_loss = descriptor_distance(
                    clip_cur,
                    clip_ref.to(device=image_tensor.device, dtype=clip_cur.dtype),
                )
                extra_loss = extra_loss + config.clip_vision_confusion_strength * clip_loss
                metrics["clip"] = clip_loss.detach()

        dino_surrogate = reference_data.get("dino_surrogate")
        dino_ref = reference_data.get("dino_protected_descriptor")
        if (
            config.dino_vision_confusion_strength > 0
            and dino_surrogate is not None
            and dino_ref is not None
            and protected_mask is not None
        ):
            dino_cur = compute_vision_surrogate_embedding(dino_surrogate, image_tensor, protected_mask)
            if dino_cur is not None:
                dino_loss = descriptor_distance(
                    dino_cur,
                    dino_ref.to(device=image_tensor.device, dtype=dino_cur.dtype),
                )
                extra_loss = extra_loss + config.dino_vision_confusion_strength * dino_loss
                metrics["dino"] = dino_loss.detach()

        face_id_surrogate = reference_data.get("face_id_surrogate")
        face_id_ref = reference_data.get("surrogate_face_id_descriptor")
        face_mask = reference_data.get("surrogate_face_mask")
        if (
            config.surrogate_face_id_strength > 0
            and face_id_surrogate is not None
            and face_id_ref is not None
            and face_mask is not None
        ):
            face_id_cur = compute_face_id_embedding(face_id_surrogate, image_tensor, face_mask)
            if face_id_cur is not None:
                face_id_loss = descriptor_distance(
                    face_id_cur,
                    face_id_ref.to(device=image_tensor.device, dtype=face_id_cur.dtype),
                )
                extra_loss = extra_loss + config.surrogate_face_id_strength * face_id_loss
                metrics["face_id"] = face_id_loss.detach()

        # --- Teacher-time purification robustness ---
        purification_views = build_teacher_purification_views(image_tensor, config)
        if purification_views:
            purification_losses = []
            if primary_mask is not None and primary_descriptor is not None:
                for view in purification_views:
                    purification_losses.append(
                        descriptor_distance(
                            compute_masked_descriptor(view, primary_mask),
                            primary_descriptor.to(device=view.device, dtype=view.dtype),
                        )
                    )
            if purification_losses:
                purification_loss = torch.stack(purification_losses).mean()
                extra_loss = extra_loss + config.teacher_purification_strength * config.reference_confusion_strength * purification_loss
                metrics["purify"] = purification_loss.detach()

            if clip_surrogate is not None and clip_ref is not None and protected_mask is not None:
                clip_pur_losses = []
                for view in purification_views:
                    clip_view = compute_vision_surrogate_embedding(clip_surrogate, view, protected_mask)
                    if clip_view is not None:
                        clip_pur_losses.append(
                            descriptor_distance(
                                clip_view,
                                clip_ref.to(device=view.device, dtype=clip_view.dtype),
                            )
                        )
                if clip_pur_losses:
                    clip_purify = torch.stack(clip_pur_losses).mean()
                    extra_loss = extra_loss + config.teacher_purification_strength * config.clip_vision_confusion_strength * clip_purify
                    metrics["clip_purify"] = clip_purify.detach()

            if dino_surrogate is not None and dino_ref is not None and protected_mask is not None:
                dino_pur_losses = []
                for view in purification_views:
                    dino_view = compute_vision_surrogate_embedding(dino_surrogate, view, protected_mask)
                    if dino_view is not None:
                        dino_pur_losses.append(
                            descriptor_distance(
                                dino_view,
                                dino_ref.to(device=view.device, dtype=dino_view.dtype),
                            )
                        )
                if dino_pur_losses:
                    dino_purify = torch.stack(dino_pur_losses).mean()
                    extra_loss = extra_loss + config.teacher_purification_strength * config.dino_vision_confusion_strength * dino_purify
                    metrics["dino_purify"] = dino_purify.detach()

            if face_id_surrogate is not None and face_id_ref is not None and face_mask is not None:
                faceid_pur_losses = []
                for view in purification_views:
                    faceid_view = compute_face_id_embedding(face_id_surrogate, view, face_mask)
                    if faceid_view is not None:
                        faceid_pur_losses.append(
                            descriptor_distance(
                                faceid_view,
                                face_id_ref.to(device=view.device, dtype=faceid_view.dtype),
                            )
                        )
                if faceid_pur_losses:
                    faceid_purify = torch.stack(faceid_pur_losses).mean()
                    extra_loss = extra_loss + config.teacher_purification_strength * config.surrogate_face_id_strength * faceid_purify
                    metrics["face_id_purify"] = faceid_purify.detach()

        # --- Weak always-on tripwires (regular profile) ---
        gray = rgb_to_luma(image_tensor)
        protected_template = reference_data.get("tripwire_protected_template")
        if config.tripwire_strength > 0 and protected_mask is not None and protected_template is not None:
            tw_align = compute_tripwire_alignment(gray, protected_template, protected_mask)
            extra_loss = extra_loss + config.tripwire_strength * tw_align
            metrics["tripwire"] = tw_align.detach()

        ring_mask = reference_data.get("tripwire_ring_mask")
        ring_template = reference_data.get("tripwire_ring_template")
        if config.tripwire_ring_strength > 0 and ring_mask is not None and ring_template is not None:
            ring_align = compute_tripwire_alignment(gray, ring_template, ring_mask)
            extra_loss = extra_loss + config.tripwire_ring_strength * ring_align
            metrics["ring"] = ring_align.detach()

        anchor_mask = reference_data.get("tripwire_anchor_mask")
        anchor_template = reference_data.get("tripwire_anchor_template")
        if config.tripwire_anchor_strength > 0 and anchor_mask is not None and anchor_template is not None:
            anchor_align = compute_tripwire_alignment(gray, anchor_template, anchor_mask)
            extra_loss = extra_loss + config.tripwire_anchor_strength * anchor_align
            metrics["anchors"] = anchor_align.detach()

        return extra_loss, metrics

    def summarize_profile(self, config):
        return {
            "profile": self.name,
            "mode": "standard",
            "tripwireStrength": float(config.tripwire_strength),
            "tripwireRingStrength": float(config.tripwire_ring_strength),
            "msDescStrength": float(config.multiscale_descriptor_strength),
            "clipStrength": float(config.clip_vision_confusion_strength),
            "dinoStrength": float(config.dino_vision_confusion_strength),
            "faceIdStrength": float(config.surrogate_face_id_strength),
            "teacherPurifyStrength": float(config.teacher_purification_strength),
        }


class SurrogateHybridImmunizationProfile(NanoBanana2ImmunizationProfile):
    name = "surrogate_hybrid"
    default_prompt = "Edit the protected subject while preserving the rest of the image."
    prompt_suffixes = (
        "{prompt} Change only the protected subject and preserve the rest of the photo.",
        "{prompt} Preserve identity while changing wardrobe, accessories, or hairstyle only where requested.",
        "{prompt} Use a reference image to keep the same person while editing the protected subject.",
        "{prompt} Clean up text, logo, or background details without changing the protected subject identity.",
        "{prompt} Swap or restyle the background while keeping the protected subject photorealistic and consistent.",
    )


class SurrogateDeeVidImmunizationProfile(StableDiffusionImmunizationProfile):
    name = "surrogate_deevid"
    default_prompt = "Edit the image realistically."
    prompt_suffixes = (
        "{prompt} Restyle the full image while keeping composition and subject placement identical.",
        "{prompt} Replace the protected subject with a different plausible subject while keeping lighting and background consistent.",
        "{prompt} Remove the protected subject and fill the region naturally while keeping the rest unchanged.",
        "{prompt} Change only the background while keeping the protected subject identical.",
        "{prompt} Use the provided reference images to guide the edit while preserving the protected subject.",
        "{prompt} Keep the protected subject photorealistic while changing clothing, accessories, or hairstyle only where requested.",
    )

    def summarize_profile(self, config):
        summary = super().summarize_profile(config)
        summary.update({
            "profile": self.name,
            "mode": "deevid_focused_surrogate",
            "clipStrength": float(config.clip_vision_confusion_strength),
            "dinoStrength": float(config.dino_vision_confusion_strength),
            "teacherPurifyStrength": float(config.teacher_purification_strength),
        })
        return summary


# ---------------------------------------------------------------------------
# NanoBanana2HardBlock — fail-closed, unchanged behavior.
# ---------------------------------------------------------------------------

class NanoBanana2HardBlockImmunizationProfile(NanoBanana2ImmunizationProfile):
    name = "nano_banana_2_hard_block"
    default_prompt = "Edit the protected subject while preserving identity, text, and logos as faithfully as possible."
    prompt_suffixes = (
        "{prompt} Keep the face, clothing, and distinctive identity details consistent from multiple references.",
        "{prompt} Edit only the protected subject while preserving facial features and subject consistency precisely.",
        "{prompt} Rewrite nearby text or logos cleanly while keeping the protected subject realistic and artifact-free.",
    )

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        reference_data = super().augment_reference_data(
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
            seed=seed,
        )
        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")
        if protected_mask is None or boundary_ring is None:
            return reference_data

        h, w = protected_mask.shape[-2:]
        # Overwrite / complement the weak templates from NanoBanana2 with the
        # full hard_block tripwire stack (stronger phase + anchor mask).
        ring_mask = normalize_map(
            boundary_ring.to(device=device, dtype=torch.float32)
            * protected_mask.to(device=device, dtype=torch.float32)
        )
        anchor_mask = build_sparse_anchor_mask(
            protected_mask.to(device=device, dtype=torch.float32),
            seed=seed,
            tile_size=config.tripwire_tile_size,
            anchor_count=config.tripwire_anchor_count,
        )
        reference_data["tripwire_protected_template"] = build_tripwire_template(
            h, w, seed=seed, device=device, phase_offset=0,
        ).detach()
        reference_data["tripwire_ring_template"] = build_tripwire_template(
            h, w, seed=(seed or 0) + 101, device=device, phase_offset=1,
        ).detach()
        reference_data["tripwire_anchor_template"] = build_tripwire_template(
            h, w, seed=(seed or 0) + 211, device=device, phase_offset=2,
        ).detach()
        reference_data["tripwire_ring_mask"] = ring_mask.detach()
        reference_data["tripwire_anchor_mask"] = anchor_mask.detach()
        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss, metrics = super().compute_extra_loss(
            transformed, reference_data, config, pipeline, dtype,
        )
        # The tripwires are already handled in NanoBanana2; hard_block just
        # runs them at full strength via the config values.
        return extra_loss, metrics

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        delta = (adversarial - base_image) * perturbation_mask
        high_frequency = delta - F.avg_pool2d(delta, kernel_size=5, stride=1, padding=2)
        hardened_delta = delta + 0.35 * high_frequency
        hardened_delta = hardened_delta.clamp(min=-config.eps * 0.9, max=config.eps * 0.9)
        hardened = base_image + 0.92 * hardened_delta
        return hardened.clamp(min=config.clamp_min, max=config.clamp_max)

    def summarize_profile(self, config):
        summary = super().summarize_profile(config)
        summary.update({
            "mode": "hard_block",
            "tripwireStrength": float(config.tripwire_strength),
            "tripwireRingStrength": float(config.tripwire_ring_strength),
            "tripwireAnchorStrength": float(config.tripwire_anchor_strength),
        })
        return summary


# ---------------------------------------------------------------------------
# NanoBanana2Distortion — fail-closed, unchanged behavior.
# ---------------------------------------------------------------------------

class NanoBanana2DistortionImmunizationProfile(NanoBanana2HardBlockImmunizationProfile):
    name = "nano_banana_2_distortion"
    default_prompt = "Edit the protected subject cleanly while preserving realism, identity consistency, and surrounding context."
    prompt_suffixes = (
        "{prompt} Produce a polished, artifact-free edit while keeping the rest of the image photorealistic.",
        "{prompt} Use strong reference consistency to keep identity details and local text/logo rendering clean.",
        "{prompt} Keep the protected subject coherent and natural-looking even after a substantial instruction-guided change.",
    )

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
        seed=None,
    ):
        reference_data = super().augment_reference_data(
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
            seed=seed,
        )
        protected_mask = reference_data.get("protected_mask")
        boundary_ring = reference_data.get("boundary_ring")
        if protected_mask is None or boundary_ring is None:
            return reference_data

        background_candidate = ((1.0 - protected_mask) * (1.0 - boundary_ring)).clamp(0, 1)
        global_anchor_mask = build_sparse_anchor_mask(
            background_candidate.to(device=device, dtype=torch.float32),
            seed=(seed or 0) + 1237,
            tile_size=config.tripwire_tile_size,
            anchor_count=config.tripwire_global_count,
        )
        if float(global_anchor_mask.mean().item()) > 0.0:
            reference_data["tripwire_global_mask"] = global_anchor_mask.detach()
            reference_data["tripwire_global_template"] = build_tripwire_template(
                protected_mask.shape[-2],
                protected_mask.shape[-1],
                seed=(seed or 0) + 313,
                device=device,
                phase_offset=3,
            ).detach()
        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        extra_loss, metrics = super().compute_extra_loss(
            transformed, reference_data, config, pipeline, dtype,
        )
        global_mask = reference_data.get("tripwire_global_mask")
        global_template = reference_data.get("tripwire_global_template")
        if config.tripwire_global_strength > 0 and global_mask is not None and global_template is not None:
            global_alignment = compute_tripwire_alignment(
                rgb_to_luma(transformed.to(dtype=torch.float32)),
                global_template,
                global_mask,
            )
            extra_loss = extra_loss + config.tripwire_global_strength * global_alignment
            metrics["global"] = global_alignment.detach()
        return extra_loss, metrics

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        delta = (adversarial - base_image) * perturbation_mask
        high_frequency = delta - F.avg_pool2d(delta, kernel_size=5, stride=1, padding=2)
        mid_frequency = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(
            delta, kernel_size=9, stride=1, padding=4,
        )
        hardened_delta = delta + 0.55 * high_frequency + 0.28 * mid_frequency
        hardened_delta = hardened_delta.clamp(min=-config.eps, max=config.eps)
        hardened = base_image + 0.98 * hardened_delta
        return hardened.clamp(min=config.clamp_min, max=config.clamp_max)

    def summarize_profile(self, config):
        summary = super().summarize_profile(config)
        summary.update({
            "mode": "distortion_block",
            "tripwireGlobalStrength": float(config.tripwire_global_strength),
        })
        return summary


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMMUNIZATION_PROFILES = {
    StableDiffusionImmunizationProfile.name: StableDiffusionImmunizationProfile(),
    FullRegenerationScaffoldImmunizationProfile.name: FullRegenerationScaffoldImmunizationProfile(),
    InstructionEditingScaffoldImmunizationProfile.name: InstructionEditingScaffoldImmunizationProfile(),
    ControlNetScaffoldImmunizationProfile.name: ControlNetScaffoldImmunizationProfile(),
    StyleTransferScaffoldImmunizationProfile.name: StyleTransferScaffoldImmunizationProfile(),
    TextAwareScaffoldImmunizationProfile.name: TextAwareScaffoldImmunizationProfile(),
    AdversarialHardenedScaffoldImmunizationProfile.name: AdversarialHardenedScaffoldImmunizationProfile(),
    NanoBananaExperimentalImmunizationProfile.name: NanoBananaExperimentalImmunizationProfile(),
    NanoBanana2ImmunizationProfile.name: NanoBanana2ImmunizationProfile(),
    SurrogateHybridImmunizationProfile.name: SurrogateHybridImmunizationProfile(),
    SurrogateDeeVidImmunizationProfile.name: SurrogateDeeVidImmunizationProfile(),
    NanoBanana2HardBlockImmunizationProfile.name: NanoBanana2HardBlockImmunizationProfile(),
    NanoBanana2DistortionImmunizationProfile.name: NanoBanana2DistortionImmunizationProfile(),
}


def get_available_immunization_profiles():
    return tuple(sorted(IMMUNIZATION_PROFILES))


def get_immunization_profile(profile_name):
    if profile_name not in IMMUNIZATION_PROFILES:
        available = ", ".join(get_available_immunization_profiles())
        raise ValueError(
            f"Unsupported immunization profile '{profile_name}'. Available profiles: {available}"
        )
    return IMMUNIZATION_PROFILES[profile_name]


# ---------------------------------------------------------------------------
# Target image builders
# ---------------------------------------------------------------------------

def _make_gray_target(size):
    return Image.new("RGB", size, color=(127, 127, 127))


def _make_noise_target(size, rng):
    noise = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(noise, mode="RGB")


def _make_checkerboard_target(size, rng):
    block = int(rng.integers(8, 65))
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    color_a = rng.integers(0, 256, size=3, dtype=np.uint8)
    color_b = rng.integers(0, 256, size=3, dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            use_a = ((x // block) + (y // block)) % 2 == 0
            arr[y, x] = color_a if use_a else color_b
    return Image.fromarray(arr, mode="RGB")


def _make_shifted_input_target(init_image, size, rng):
    target = resize_and_crop(init_image.copy(), size)
    if rng.random() < 0.5:
        target = ImageOps.mirror(target)
    if rng.random() < 0.5:
        target = ImageOps.flip(target)
    if rng.random() < 0.75:
        target = target.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.0, 10.0))))
    return target


def build_target_image(init_image, mode="random", size=(512, 512), seed=None):
    rng = np.random.default_rng(seed)
    if mode == "gray":
        return _make_gray_target(size)
    if mode == "noise":
        return _make_noise_target(size, rng)
    if mode == "checkerboard":
        return _make_checkerboard_target(size, rng)
    if mode == "shifted_input":
        return _make_shifted_input_target(init_image, size, random.Random(seed))
    if mode == "random":
        modes = ("noise", "checkerboard", "shifted_input")
        picked_mode = random.Random(seed).choice(modes)
        return build_target_image(init_image, mode=picked_mode, size=size, seed=seed)
    raise ValueError(f"Unsupported target mode: {mode}")


# ---------------------------------------------------------------------------
# Encoding / latent helpers
# ---------------------------------------------------------------------------

def encode_prompt_variant(pipeline, prompt, device, do_classifier_free_guidance):
    encode_prompt_params = inspect.signature(pipeline.encode_prompt).parameters
    encode_kwargs = {
        "prompt": prompt,
        "device": device,
        "num_images_per_prompt": 1,
        "do_classifier_free_guidance": do_classifier_free_guidance,
        "negative_prompt": "",
    }
    if is_sdxl_pipeline(pipeline):
        if "prompt_2" in encode_prompt_params:
            encode_kwargs["prompt_2"] = prompt
        if "negative_prompt_2" in encode_prompt_params:
            encode_kwargs["negative_prompt_2"] = ""

    encode_kwargs = {
        key: value for key, value in encode_kwargs.items()
        if key in encode_prompt_params
    }
    encoded = pipeline.encode_prompt(**encode_kwargs)

    if is_sdxl_pipeline(pipeline):
        if not isinstance(encoded, tuple) or len(encoded) < 4:
            raise RuntimeError("Unexpected SDXL prompt encoding payload.")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded[:4]
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    if not isinstance(encoded, tuple) or len(encoded) < 2:
        raise RuntimeError("Unexpected Stable Diffusion prompt encoding payload.")
    prompt_embeds, negative_prompt_embeds = encoded[:2]
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return {"prompt_embeds": prompt_embeds}


def encode_target(vae, target_image, device, dtype):
    with torch.no_grad():
        target_tensor = preprocess(target_image).to(device=device, dtype=dtype)
        return vae.encode(target_tensor).latent_dist.mean.detach()


def encode_image_latents(vae, image_tensor):
    latents = vae.encode(image_tensor).latent_dist.mean
    return vae.config.scaling_factor * latents


def duplicate_for_cfg(tensor, do_classifier_free_guidance):
    return torch.cat([tensor, tensor], dim=0) if do_classifier_free_guidance else tensor


def prepare_mask_latents(mask_tensor, dtype, device, do_classifier_free_guidance):
    latents = F.interpolate(mask_tensor, size=(mask_tensor.shape[-2] // 8, mask_tensor.shape[-1] // 8))
    latents = latents.to(device=device, dtype=dtype)
    return duplicate_for_cfg(latents, do_classifier_free_guidance)


# ---------------------------------------------------------------------------
# Spatial mask builders
# ---------------------------------------------------------------------------

def build_boundary_ring(mask_tensor, ring_width):
    if ring_width <= 0:
        return torch.zeros_like(mask_tensor)
    kernel_size = ring_width * 2 + 1
    dilated = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=ring_width)
    eroded = 1 - F.max_pool2d(1 - mask_tensor, kernel_size=kernel_size, stride=1, padding=ring_width)
    return (dilated - eroded).clamp(0, 1)


def normalize_map(tensor):
    maximum = tensor.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (tensor / maximum).clamp(0, 1)


def rgb_to_luma(tensor):
    return 0.2989 * tensor[:, 0:1] + 0.5870 * tensor[:, 1:2] + 0.1140 * tensor[:, 2:3]


def sobel_magnitude(gray_tensor):
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=gray_tensor.device, dtype=gray_tensor.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=gray_tensor.device, dtype=gray_tensor.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(gray_tensor, kernel_x, padding=1)
    grad_y = F.conv2d(gray_tensor, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def build_region_priority_maps(init_image, protected_mask, boundary_ring, config, device):
    image_array = np.asarray(init_image.convert("RGB"), dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    protected_mask = protected_mask.to(device=device, dtype=torch.float32)
    boundary_ring = boundary_ring.to(device=device, dtype=torch.float32)

    red = image_tensor[:, 0:1]
    green = image_tensor[:, 1:2]
    blue = image_tensor[:, 2:3]
    max_rgb = image_tensor.amax(dim=1, keepdim=True)
    min_rgb = image_tensor.amin(dim=1, keepdim=True)
    saturation = (max_rgb - min_rgb).clamp(0, 1)
    gray = rgb_to_luma(image_tensor)
    edge_map = normalize_map(sobel_magnitude(gray))

    skin_map = (
        (red > 0.35) & (green > 0.2) & (blue > 0.15)
        & ((max_rgb - min_rgb) > 0.08)
        & (red > green) & (red > blue)
    ).to(dtype=torch.float32)
    skin_map = normalize_map(F.avg_pool2d(skin_map, kernel_size=11, stride=1, padding=5))
    face_proxy_map = normalize_map(F.max_pool2d(skin_map, kernel_size=21, stride=1, padding=10))
    text_anchor_map = normalize_map(edge_map * (1.0 - saturation))
    logo_anchor_map = normalize_map(edge_map * (0.35 + saturation))

    priority_map = protected_mask + 0.5 * boundary_ring
    if config.priority_face_strength > 0:
        priority_map = priority_map + config.priority_face_strength * face_proxy_map * protected_mask
    if config.priority_skin_strength > 0:
        priority_map = priority_map + config.priority_skin_strength * skin_map * protected_mask
    if config.priority_text_strength > 0:
        priority_map = priority_map + config.priority_text_strength * text_anchor_map * protected_mask
    if config.priority_logo_strength > 0:
        priority_map = priority_map + config.priority_logo_strength * logo_anchor_map * protected_mask

    return {
        "region_priority_map": priority_map.clamp(0, 1).detach(),
        "face_proxy_map": face_proxy_map.detach(),
        "skin_proxy_map": skin_map.detach(),
        "text_anchor_map": text_anchor_map.detach(),
        "logo_anchor_map": logo_anchor_map.detach(),
    }


# ---------------------------------------------------------------------------
# Frequency / watermark helpers
# ---------------------------------------------------------------------------

def build_frequency_band_mask(height, width, band_low, band_high, device):
    safe_low = max(0.0, float(band_low))
    safe_high = max(safe_low + 1e-4, min(0.5, float(band_high)))
    freq_y = torch.fft.fftfreq(height, device=device).reshape(-1, 1)
    freq_x = torch.fft.fftfreq(width, device=device).reshape(1, -1)
    radius = torch.sqrt(freq_x.square() + freq_y.square())
    return ((radius >= safe_low) & (radius <= safe_high)).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def build_band_limited_noise(height, width, band_mask, strength, device):
    noise = torch.randn((1, 1, height, width), device=device, dtype=torch.float32)
    filtered = torch.fft.ifft2(torch.fft.fft2(noise) * band_mask).real
    filtered = filtered / (filtered.std().clamp_min(1e-6))
    return filtered * float(strength)


def build_watermark_template(height, width, seed, band_low, band_high, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed or 0) + height * 37 + width * 17)
    noise = torch.randn((1, 1, height, width), generator=generator, device=device, dtype=torch.float32)
    band_mask = build_frequency_band_mask(height, width, band_low, band_high, device)
    watermark = torch.fft.ifft2(torch.fft.fft2(noise) * band_mask).real
    return watermark / watermark.std().clamp_min(1e-6)


def compute_watermark_alignment(transformed, band_mask, watermark_template, spatial_mask):
    gray = rgb_to_luma(transformed.to(dtype=torch.float32))
    response = torch.fft.ifft2(torch.fft.fft2(gray) * band_mask).real
    if spatial_mask is not None:
        mask = spatial_mask.to(dtype=torch.float32)
        response = response * mask
        watermark_template = watermark_template * mask
        denom = mask.mean().clamp_min(1e-6)
    else:
        denom = response.new_tensor(1.0)
    return (response * watermark_template).mean().abs() / denom


# ---------------------------------------------------------------------------
# Jitter / compression / multi-view transform helpers
# ---------------------------------------------------------------------------

def ste_round(tensor):
    return tensor + (torch.round(tensor) - tensor).detach()


def apply_subpixel_translation(tensor, max_shift_pixels):
    if max_shift_pixels <= 0:
        return tensor
    shift_x = float(torch.empty(1, device=tensor.device).uniform_(-max_shift_pixels, max_shift_pixels).item())
    shift_y = float(torch.empty(1, device=tensor.device).uniform_(-max_shift_pixels, max_shift_pixels).item())
    translate_x = 2.0 * shift_x / max(tensor.shape[-1], 1)
    translate_y = 2.0 * shift_y / max(tensor.shape[-2], 1)
    theta = torch.tensor(
        [[[1.0, 0.0, translate_x], [0.0, 1.0, translate_y]]],
        device=tensor.device, dtype=tensor.dtype,
    ).expand(tensor.shape[0], -1, -1)
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_light_compression_proxy(tensor, strength):
    if strength <= 0:
        return tensor
    pooled = F.avg_pool2d(tensor, kernel_size=2, stride=2)
    restored = F.interpolate(pooled, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    quant_step = 0.015 + 0.08 * float(strength)
    quantized = ste_round(restored / quant_step) * quant_step
    blend = min(0.45, 3.5 * float(strength))
    return tensor.lerp(quantized, blend)


def apply_updown_scale_proxy(tensor, scale=0.75):
    """Downscale then upscale — simulates an upscaler pipeline pass.
    The trip down and back smears fine perturbation detail; training against
    this makes the signal more robust to resolution-change pipelines."""
    if scale <= 0 or scale >= 1.0:
        return tensor
    h, w = tensor.shape[-2:]
    nh, nw = max(32, int(h * scale)), max(32, int(w * scale))
    down = F.interpolate(tensor, size=(nh, nw), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def apply_sharpen_proxy(tensor, strength=0.3):
    """Cheap unsharp-mask: simulates a deartifact/sharpen post-processing step.
    Optimizing through this keeps high-frequency signal alive after such passes."""
    blurred = F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    sharpened = tensor + strength * (tensor - blurred)
    return sharpened.clamp(tensor.min(), tensor.max())


def compute_high_frequency_change(transformed, original, mask):
    transformed_hp = transformed - F.avg_pool2d(transformed, kernel_size=5, stride=1, padding=2)
    original_hp = original - F.avg_pool2d(original, kernel_size=5, stride=1, padding=2)
    return ((transformed_hp - original_hp).abs() * mask).mean()


def compute_edge_change(transformed, original, mask):
    transformed_edges = sobel_magnitude(rgb_to_luma(transformed.to(dtype=torch.float32)))
    original_edges = sobel_magnitude(rgb_to_luma(original.to(dtype=torch.float32)))
    return ((transformed_edges - original_edges).abs() * mask.to(dtype=torch.float32)).mean()


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------

def compute_masked_descriptor(image_tensor, spatial_mask):
    mask = spatial_mask.to(device=image_tensor.device, dtype=torch.float32)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    if mask.shape[-2:] != image_tensor.shape[-2:]:
        mask = F.interpolate(mask, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    rgb_mask = mask.repeat(1, image_tensor.shape[1], 1, 1)
    norm = mask.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)

    low_frequency = F.adaptive_avg_pool2d(image_tensor.to(dtype=torch.float32) * rgb_mask, output_size=(4, 4)) / norm
    edge_map = sobel_magnitude(rgb_to_luma(image_tensor.to(dtype=torch.float32)))
    edge_features = F.adaptive_avg_pool2d(edge_map * mask, output_size=(4, 4)) / norm
    high_frequency = image_tensor.to(dtype=torch.float32) - F.avg_pool2d(
        image_tensor.to(dtype=torch.float32), kernel_size=5, stride=1, padding=2,
    )
    high_frequency = high_frequency.abs().mean(dim=1, keepdim=True)
    high_frequency_features = F.adaptive_avg_pool2d(high_frequency * mask, output_size=(4, 4)) / norm

    descriptor = torch.cat(
        [low_frequency.flatten(start_dim=1), edge_features.flatten(start_dim=1), high_frequency_features.flatten(start_dim=1)],
        dim=1,
    )
    return F.normalize(descriptor, dim=1)


def compute_multiscale_descriptor(image_tensor, spatial_mask, scales=(1.0, 0.5)):
    """Concatenate single-scale descriptors at each requested resolution.
    Reference-guided editors (e.g. FLUX Kontext, IP-Adapter) embed features at
    multiple spatial scales; mismatching the fingerprint at all scales
    simultaneously maximises confusion across encoder variants."""
    parts = []
    for scale in scales:
        if abs(scale - 1.0) < 1e-4:
            img, msk = image_tensor, spatial_mask
        else:
            h, w = image_tensor.shape[-2:]
            nh, nw = max(32, int(h * scale)), max(32, int(w * scale))
            img = F.interpolate(image_tensor, size=(nh, nw), mode="bilinear", align_corners=False)
            msk = F.interpolate(spatial_mask, size=(nh, nw), mode="bilinear", align_corners=False)
        parts.append(compute_masked_descriptor(img, msk))
    return torch.cat(parts, dim=1)


def descriptor_distance(left, right):
    return (left - right).abs().mean()


def descriptor_similarity(left, right):
    return F.cosine_similarity(left, right, dim=1).mean()


def build_identity_drift_view(image_tensor, config):
    drift_view = apply_subpixel_translation(
        image_tensor, max(0.08, float(config.subpixel_jitter) * 0.65),
    )
    drift_view = apply_light_compression_proxy(
        drift_view, max(0.01, float(config.compression_jitter_strength) * 0.6),
    )
    if config.frequency_noise_strength > 0:
        h, w = image_tensor.shape[-2:]
        band_mask = build_frequency_band_mask(h, w, config.frequency_band_low, config.frequency_band_high, image_tensor.device)
        band_noise = build_band_limited_noise(
            h, w, band_mask, max(0.0008, float(config.frequency_noise_strength) * 0.35), image_tensor.device,
        ).to(device=image_tensor.device, dtype=image_tensor.dtype)
        drift_view = drift_view + band_noise.repeat(1, image_tensor.shape[1], 1, 1)
    return drift_view.clamp(min=config.clamp_min, max=config.clamp_max)


def build_teacher_purification_views(image_tensor, config):
    strength = float(max(0.0, getattr(config, "teacher_purification_strength", 0.0)))
    count = int(max(0, getattr(config, "teacher_purification_view_count", 0)))
    if strength <= 0.0 or count <= 0:
        return []

    views = []
    compression_strength = max(0.01, strength * 0.8)
    updown_scale = max(0.55, 1.0 - min(0.4, strength * 0.5))
    sharpen_strength = min(0.45, 0.12 + strength * 0.6)

    view = apply_light_compression_proxy(image_tensor, compression_strength).clamp(
        min=config.clamp_min,
        max=config.clamp_max,
    )
    views.append(view)
    if count == 1:
        return views

    view = apply_updown_scale_proxy(view, scale=updown_scale).clamp(
        min=config.clamp_min,
        max=config.clamp_max,
    )
    views.append(view)
    if count == 2:
        return views

    view = apply_sharpen_proxy(view, strength=sharpen_strength).clamp(
        min=config.clamp_min,
        max=config.clamp_max,
    )
    views.append(view)
    return views[:count]


# ---------------------------------------------------------------------------
# Tripwire helpers
# ---------------------------------------------------------------------------

def build_tripwire_template(height, width, seed, device, phase_offset=0):
    rng = random.Random((seed or 0) + 104729 * int(phase_offset))
    x = torch.linspace(-math.pi, math.pi, width, device=device, dtype=torch.float32).view(1, 1, 1, width)
    y = torch.linspace(-math.pi, math.pi, height, device=device, dtype=torch.float32).view(1, 1, height, 1)
    template = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    for _ in range(3):
        angle = rng.uniform(0.0, math.pi)
        projection = math.cos(angle) * x + math.sin(angle) * y
        frequency = rng.uniform(8.0, 18.0)
        phase = rng.uniform(-math.pi, math.pi)
        template = template + torch.sin(frequency * projection + phase)
    checker = torch.sin(rng.uniform(6.0, 11.0) * x + rng.uniform(-math.pi, math.pi))
    checker = checker * torch.cos(rng.uniform(6.0, 11.0) * y + rng.uniform(-math.pi, math.pi))
    template = template + 0.5 * checker
    return template / template.std().clamp_min(1e-6)


def build_sparse_anchor_mask(source_mask, seed, tile_size=32, anchor_count=4):
    if anchor_count <= 0:
        return torch.zeros_like(source_mask, dtype=torch.float32)
    mask = source_mask.to(dtype=torch.float32)
    height, width = mask.shape[-2:]
    tile = max(8, int(tile_size))
    candidates = []
    for top in range(0, max(1, height - tile + 1), tile):
        for left in range(0, max(1, width - tile + 1), tile):
            patch = mask[..., top: min(top + tile, height), left: min(left + tile, width)]
            if float(patch.mean().item()) > 0.65:
                candidates.append((top, left))
    rng = random.Random((seed or 0) + 7919)
    rng.shuffle(candidates)
    anchor_mask = torch.zeros_like(mask, dtype=torch.float32)
    for top, left in candidates[:anchor_count]:
        anchor_mask[..., top: min(top + tile, height), left: min(left + tile, width)] = 1.0
    return anchor_mask


def compute_tripwire_alignment(gray_tensor, template, spatial_mask):
    mask = spatial_mask.to(device=gray_tensor.device, dtype=torch.float32)
    if mask.shape[-2:] != gray_tensor.shape[-2:]:
        mask = F.interpolate(mask, size=gray_tensor.shape[-2:], mode="bilinear", align_corners=False)
    if template.shape[-2:] != gray_tensor.shape[-2:]:
        template = F.interpolate(template, size=gray_tensor.shape[-2:], mode="bilinear", align_corners=False)
    masked_mean = (gray_tensor * mask).sum(dim=(-2, -1), keepdim=True) / mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    centered = (gray_tensor - masked_mean) * mask
    return (centered * template).mean().abs() / mask.mean().clamp_min(1e-4)


# ---------------------------------------------------------------------------
# Reference region mask builder
# ---------------------------------------------------------------------------

def build_reference_region_masks(reference_data, config):
    protected_mask = reference_data.get("protected_mask")
    if protected_mask is None:
        return []
    selected_masks = [protected_mask.to(dtype=torch.float32)]
    candidates = []
    for key in ("face_proxy_map", "skin_proxy_map", "text_anchor_map", "logo_anchor_map"):
        proxy_map = reference_data.get(key)
        if proxy_map is None:
            continue
        candidate = normalize_map(proxy_map.to(dtype=torch.float32) * protected_mask.to(dtype=torch.float32))
        coverage = float(candidate.mean().item())
        if coverage > 0.005:
            candidates.append((coverage, candidate))
    candidates.sort(key=lambda item: item[0], reverse=True)
    max_regions = max(1, int(config.reference_region_count))
    for _, candidate in candidates[: max(0, max_regions - 1)]:
        selected_masks.append(candidate)
    return selected_masks


# ---------------------------------------------------------------------------
# PGD optimizer
# ---------------------------------------------------------------------------

def select_proxy_timesteps(timesteps, denoiser_steps):
    if denoiser_steps <= 0:
        return []
    if len(timesteps) <= denoiser_steps:
        return [t.reshape(1) for t in timesteps]
    indexes = torch.linspace(0, len(timesteps) - 1, steps=denoiser_steps).round().long().tolist()
    return [timesteps[i].reshape(1) for i in indexes]


def predict_guided_noise(
    pipeline,
    scheduler,
    image_latents,
    mask_latents,
    masked_image_latents,
    prompt_conditioning,
    add_time_ids,
    timestep,
    guidance_scale,
    do_classifier_free_guidance,
    noise,
):
    timestep = timestep.to(device=image_latents.device)
    noisy_latents = scheduler.add_noise(image_latents, noise, timestep)
    latent_model_input = duplicate_for_cfg(noisy_latents, do_classifier_free_guidance)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    latent_model_input = torch.cat([latent_model_input, mask_latents, masked_image_latents], dim=1)
    prompt_embeds = prompt_conditioning["prompt_embeds"]
    added_cond_kwargs = None
    pooled_prompt_embeds = prompt_conditioning.get("pooled_prompt_embeds")
    if pooled_prompt_embeds is not None:
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds}
        if add_time_ids is not None:
            added_cond_kwargs["time_ids"] = add_time_ids
    noise_pred = pipeline.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def apply_random_view(X_adv, base_image, perturbation_mask, config):
    if (
        config.resize_jitter <= 0
        and config.noise_strength <= 0
        and config.blur_kernel_size <= 1
        and config.subpixel_jitter <= 0
        and config.compression_jitter_strength <= 0
        and config.frequency_noise_strength <= 0
        and config.updown_scale_jitter <= 0
        and config.sharpen_proxy_strength <= 0
    ):
        return X_adv

    height, width = X_adv.shape[-2:]
    min_scale = max(0.5, 1.0 - config.resize_jitter)
    max_scale = 1.0 + config.resize_jitter
    scale = float(torch.empty(1, device=X_adv.device).uniform_(min_scale, max_scale).item())
    resized_h = max(32, int(round(height * scale / 32) * 32))
    resized_w = max(32, int(round(width * scale / 32) * 32))

    transformed = F.interpolate(X_adv, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    transformed = F.interpolate(transformed, size=(height, width), mode="bilinear", align_corners=False)
    transformed = apply_subpixel_translation(transformed, config.subpixel_jitter)
    transformed = apply_light_compression_proxy(transformed, config.compression_jitter_strength)

    # Updown scale proxy — simulates upscaler pipeline (nano_banana_2+).
    if config.updown_scale_jitter > 0 and torch.rand(1, device=X_adv.device).item() < 0.5:
        transformed = apply_updown_scale_proxy(transformed, scale=1.0 - config.updown_scale_jitter)

    # Cheap unsharp-mask sharpen proxy (nano_banana_2+).
    if config.sharpen_proxy_strength > 0 and torch.rand(1, device=X_adv.device).item() < 0.4:
        transformed = apply_sharpen_proxy(transformed, strength=config.sharpen_proxy_strength)

    if config.blur_kernel_size > 1 and torch.rand(1, device=X_adv.device).item() < 0.5:
        kernel_size = config.blur_kernel_size
        transformed = F.avg_pool2d(transformed, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    if config.noise_strength > 0:
        noise = torch.randn_like(transformed) * config.noise_strength
        transformed = transformed + noise * perturbation_mask

    if config.frequency_noise_strength > 0:
        band_mask = build_frequency_band_mask(
            height, width, config.frequency_band_low, config.frequency_band_high, transformed.device,
        )
        band_noise = build_band_limited_noise(
            height, width, band_mask, config.frequency_noise_strength, transformed.device,
        ).to(device=transformed.device, dtype=transformed.dtype)
        transformed = transformed + band_noise.repeat(1, transformed.shape[1], 1, 1) * perturbation_mask

    transformed = transformed.clamp(min=config.clamp_min, max=config.clamp_max)
    return base_image + (transformed - base_image) * perturbation_mask


def pgd_optimize(
    X,
    profile,
    reference_data,
    pipeline,
    config,
    *,
    eps=0.1,
    step_size=0.015,
    iters=40,
    clamp_min=0,
    clamp_max=1,
    mask=None,
    progress_callback: ProgressCallback | None = None,
):
    X_adv = X.clone().detach() + (torch.rand_like(X) * 2 * eps - eps)
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv.requires_grad_(True)
        eot_count = max(1, config.eot_samples)
        accumulated_grad = torch.zeros_like(X_adv, dtype=torch.float32)
        loss_total = 0.0
        metric_totals: dict[str, float] = {}
        for _ in range(eot_count):
            transformed = apply_random_view(X_adv, X, mask, config)
            total_loss, metrics = profile.compute_loss(
                transformed, reference_data, config, pipeline, transformed.dtype,
            )
            scaled_loss = total_loss / float(eot_count)
            grad, = torch.autograd.grad(scaled_loss, [X_adv], retain_graph=False, create_graph=False)
            accumulated_grad.add_(grad.detach().to(dtype=torch.float32))
            loss_total += float(total_loss.detach().item())
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + float(value.detach().item())
            del transformed, total_loss, scaled_loss, grad, metrics

        loss_value = loss_total / float(eot_count)
        metric_means = {
            name: value / float(eot_count)
            for name, value in metric_totals.items()
        }
        metric_summary = " | ".join(
            f"{name} {value:.3f}"
            for name, value in metric_means.items()
        )
        description = f"Loss {loss_value:.5f} | step {actual_step_size:.4}"
        if metric_summary:
            description = f"Loss {loss_value:.5f} | {metric_summary} | step {actual_step_size:.4}"
        pbar.set_description(description)

        if progress_callback is not None:
            progress_callback({
                "iteration": i + 1,
                "totalIterations": iters,
                "loss": loss_value,
                "stepSize": float(actual_step_size),
                "metrics": metric_means,
            })

        X_adv = X_adv - accumulated_grad.sign().to(dtype=X_adv.dtype) * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv = X_adv.clamp(min=clamp_min, max=clamp_max).detach()
        if mask is not None:
            X_adv = X + (X_adv - X) * mask

    return X_adv


# ---------------------------------------------------------------------------
# Low-memory fallback helper
# ---------------------------------------------------------------------------

def _scale_config_for_low_memory(config, profile_name):
    """Scale profile-specific losses by low_memory_loss_scale instead of
    zeroing them.  Fail-closed profiles (hard_block, distortion) still zero."""
    if not getattr(config, "allow_low_memory_fallback", True):
        return config
    if profile_name in ("nano_banana_2_hard_block", "nano_banana_2_distortion"):
        # Fail-closed: preserve existing behaviour (caller should raise OOM).
        return config
    scale = max(0.0, float(config.low_memory_loss_scale))
    return replace(
        config,
        iters=max(6, min(config.iters, int(round(config.iters * 0.55)))),
        step_size=min(config.step_size, max(config.step_size * 0.9, config.step_size / 2)),
        denoiser_strength=config.denoiser_strength * min(1.0, max(0.0, scale)),
        denoiser_steps=min(config.denoiser_steps, 1),
        eot_samples=1,
        resize_jitter=min(config.resize_jitter, 0.04),
        noise_strength=min(config.noise_strength, 0.008),
        blur_kernel_size=min(config.blur_kernel_size, 3),
        compression_jitter_strength=min(config.compression_jitter_strength, 0.03),
        subpixel_jitter=min(config.subpixel_jitter, 0.18),
        frequency_noise_strength=min(config.frequency_noise_strength, 0.002),
        max_prompt_variants=1 if config.max_prompt_variants <= 1 else min(config.max_prompt_variants, 2),
        reference_region_count=min(config.reference_region_count, 2),
        background_anchor_count=min(config.background_anchor_count, 2),
        tripwire_anchor_count=min(config.tripwire_anchor_count, 2),
        tripwire_global_count=min(config.tripwire_global_count, 1),
        multiscale_descriptor_scales=(1.0,),
        reference_confusion_strength=config.reference_confusion_strength * scale,
        identity_drift_strength=config.identity_drift_strength * scale,
        surrogate_face_id_strength=config.surrogate_face_id_strength * scale,
        clip_vision_confusion_strength=config.clip_vision_confusion_strength * scale,
        dino_vision_confusion_strength=config.dino_vision_confusion_strength * scale,
        context_blend_strength=config.context_blend_strength * scale,
        multiscale_descriptor_strength=config.multiscale_descriptor_strength * scale,
        background_anchor_confusion_strength=config.background_anchor_confusion_strength * scale,
        tripwire_strength=config.tripwire_strength * scale,
        tripwire_ring_strength=config.tripwire_ring_strength * scale,
        tripwire_anchor_strength=config.tripwire_anchor_strength * scale,
        tripwire_global_strength=config.tripwire_global_strength * scale,
        watermark_strength=config.watermark_strength * scale,
        region_priority_strength=config.region_priority_strength * scale,
        semantic_boundary_strength=config.semantic_boundary_strength * scale,
        teacher_purification_strength=config.teacher_purification_strength * scale,
        teacher_purification_view_count=min(config.teacher_purification_view_count, 1),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def immunize_image(
    init_image,
    mask_image,
    pipeline,
    prompt="",
    guidance_scale=7.5,
    num_inference_steps=25,
    config=None,
    seed=None,
    progress_callback: ProgressCallback | None = None,
):
    config = config or ImmunizationConfig()
    profile = get_immunization_profile(config.profile_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    mask, masked_image = prepare_mask_and_masked_image(init_image, mask_image)
    X = masked_image.to(device=device, dtype=dtype)
    base_image = X
    mask = mask.to(device=device, dtype=dtype)
    perturbation_mask = 1 - mask

    reference_data = profile.prepare_reference_data(
        init_image,
        mask,
        X,
        pipeline,
        prompt,
        guidance_scale,
        num_inference_steps,
        config,
        device,
        dtype,
        seed=seed,
    )

    active_config = config
    try:
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            adv_X = pgd_optimize(
                X,
                profile=profile,
                reference_data=reference_data,
                pipeline=pipeline,
                config=active_config,
                eps=active_config.eps,
                step_size=active_config.step_size,
                iters=active_config.iters,
                clamp_min=active_config.clamp_min,
                clamp_max=active_config.clamp_max,
                mask=perturbation_mask,
                progress_callback=progress_callback,
            )
    except torch.cuda.OutOfMemoryError:
        # Scale down losses instead of zeroing. Strict profiles still fail closed.
        torch.cuda.empty_cache()
        if not getattr(config, "allow_low_memory_fallback", True):
            raise
        active_config = _scale_config_for_low_memory(config, config.profile_name)
        if active_config is config:
            raise
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            adv_X = pgd_optimize(
                X,
                profile=profile,
                reference_data=reference_data,
                pipeline=pipeline,
                config=active_config,
                eps=active_config.eps,
                step_size=active_config.step_size,
                iters=active_config.iters,
                clamp_min=active_config.clamp_min,
                clamp_max=active_config.clamp_max,
                mask=perturbation_mask,
                progress_callback=progress_callback,
            )
    finally:
        try:
            del X, mask, masked_image
        except Exception:
            pass

    adv_X = profile.finalize_adversarial(adv_X, base_image, perturbation_mask, active_config)
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    adv_image = topil(adv_X[0].detach().cpu()).convert("RGB")
    adv_image = recover_image(adv_image, init_image, mask_image, background=True)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return adv_image, reference_data.get("target_image")
