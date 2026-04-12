from dataclasses import dataclass
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from .utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop

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
    eot_samples: int = 4
    resize_jitter: float = 0.15
    noise_strength: float = 0.03
    blur_kernel_size: int = 5
    semantic_boundary_strength: float = 0.0
    semantic_ring_width: int = 9
    max_prompt_variants: int = 0


class StableDiffusionImmunizationProfile:
    name = "stable_diffusion"

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
    ):
        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        return transformed.new_tensor(0.0), {}

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        return adversarial

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

        original_latents = encode_image_latents(
            pipeline.vae,
            preprocess(init_image).to(device=device, dtype=dtype),
        )
        masked_image_latents = encode_image_latents(pipeline.vae, masked_image)
        mask_latents = prepare_mask_latents(mask_tensor, dtype, device, do_classifier_free_guidance)
        masked_image_latents_cfg = duplicate_for_cfg(masked_image_latents, do_classifier_free_guidance)

        denoiser_timesteps = select_proxy_timesteps(scheduler.timesteps, config.denoiser_steps)
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
                    step_predictions.append(
                        predict_guided_noise(
                            pipeline,
                            scheduler,
                            original_latents,
                            mask_latents,
                            masked_image_latents_cfg,
                            prompt_embeds,
                            timestep,
                            guidance_scale,
                            do_classifier_free_guidance,
                            noise,
                        ).detach()
                    )
                original_noise_predictions.append(step_predictions)

        reference_data = {
            "target_image": target_image,
            "targets": encode_target(pipeline.vae, target_image, device, dtype),
            "original_latents": original_latents.detach(),
            "masked_image": masked_image.detach(),
            "mask_tensor": mask_tensor.detach(),
            "mask_latents": mask_latents.detach(),
            "prompt_embeds_variants": [prompt_embeds.detach() for prompt_embeds in prompt_embeds_variants],
            "guidance_scale": guidance_scale,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "scheduler": scheduler,
            "denoiser_timesteps": denoiser_timesteps,
            "denoiser_noises": noises,
            "original_noise_predictions": original_noise_predictions,
        }
        return self.augment_reference_data(
            reference_data,
            init_image,
            mask_tensor,
            masked_image,
            config,
            device,
            dtype,
        )

    def compute_loss(self, transformed, reference_data, config, pipeline, dtype):
        transformed_latents = encode_image_latents(pipeline.vae, transformed)
        target_loss = (transformed_latents - reference_data["targets"]).norm()
        chaos_loss = (transformed_latents - reference_data["original_latents"]).norm()
        denoiser_loss = transformed_latents.new_tensor(0.0)
        if config.denoiser_strength > 0 and reference_data["denoiser_timesteps"]:
            transformed_masked_latents = encode_image_latents(pipeline.vae, transformed)
            transformed_masked_latents = duplicate_for_cfg(
                transformed_masked_latents,
                reference_data["do_classifier_free_guidance"],
            )
            denoiser_losses = []
            for timestep, noise, original_noise_predictions in zip(
                reference_data["denoiser_timesteps"],
                reference_data["denoiser_noises"],
                reference_data["original_noise_predictions"],
            ):
                for prompt_embeds, original_noise_prediction in zip(
                    reference_data["prompt_embeds_variants"],
                    original_noise_predictions,
                ):
                    guided_noise = predict_guided_noise(
                        pipeline,
                        reference_data["scheduler"],
                        transformed_latents,
                        reference_data["mask_latents"],
                        transformed_masked_latents,
                        prompt_embeds,
                        timestep,
                        reference_data["guidance_scale"],
                        reference_data["do_classifier_free_guidance"],
                        noise,
                    )
                    denoiser_losses.append((guided_noise - original_noise_prediction).norm())
            denoiser_loss = torch.stack(denoiser_losses).mean()

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


class NanoBananaExperimentalImmunizationProfile(StableDiffusionImmunizationProfile):
    name = "nano_banana_experimental"

    def build_prompt_variants(self, prompt):
        base_prompt = (prompt or "").strip()
        seed_prompt = base_prompt or "Edit the protected subject while preserving the rest of the image."
        candidates = [
            seed_prompt,
            f"{seed_prompt} Change only the protected subject and preserve the rest of the photo.",
            f"{seed_prompt} Replace the protected subject with a different plausible subject while keeping the lighting and background consistent.",
            f"{seed_prompt} Remove the protected subject and fill the region realistically while preserving the rest of the image.",
        ]

        prompt_variants = []
        seen = set()
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                prompt_variants.append(normalized)
        return prompt_variants

    def augment_reference_data(
        self,
        reference_data,
        init_image,
        mask_tensor,
        masked_image,
        config,
        device,
        dtype,
    ):
        reference_data["boundary_ring"] = build_boundary_ring(mask_tensor.detach(), config.semantic_ring_width).detach()
        reference_data["protected_mask"] = (1 - mask_tensor).detach()
        reference_data["original_masked_image"] = masked_image.detach()
        return reference_data

    def compute_extra_loss(self, transformed, reference_data, config, pipeline, dtype):
        if config.semantic_boundary_strength <= 0:
            return transformed.new_tensor(0.0), {}

        boundary_ring = reference_data["boundary_ring"]
        protected_mask = reference_data["protected_mask"]
        original_masked_image = reference_data["original_masked_image"]

        boundary_change = ((transformed - original_masked_image).abs() * boundary_ring).mean()
        protected_change = ((transformed - original_masked_image).abs() * protected_mask).mean()
        semantic_loss = config.semantic_boundary_strength * (boundary_change + 0.25 * protected_change)
        metrics = {
            "semantic": boundary_change.detach(),
            "protected": protected_change.detach(),
        }
        return semantic_loss, metrics

    def finalize_adversarial(self, adversarial, base_image, perturbation_mask, config):
        delta = (adversarial - base_image) * perturbation_mask
        if config.blur_kernel_size > 1:
            delta = F.avg_pool2d(
                delta,
                kernel_size=config.blur_kernel_size,
                stride=1,
                padding=config.blur_kernel_size // 2,
            )

        # Keep the experimental defense visually closer to the source image now that
        # the API preserves higher-fidelity outputs instead of hiding artifacts behind
        # lossy compression and aggressive downscaling.
        delta = delta.clamp(min=-config.eps * 0.6, max=config.eps * 0.6)
        smoothed = base_image + (0.65 * delta)
        return smoothed.clamp(min=config.clamp_min, max=config.clamp_max)


IMMUNIZATION_PROFILES = {
    StableDiffusionImmunizationProfile.name: StableDiffusionImmunizationProfile(),
    NanoBananaExperimentalImmunizationProfile.name: NanoBananaExperimentalImmunizationProfile(),
}


def get_available_immunization_profiles():
    return tuple(sorted(IMMUNIZATION_PROFILES))


def get_immunization_profile(profile_name):
    if profile_name not in IMMUNIZATION_PROFILES:
        available = ", ".join(get_available_immunization_profiles())
        raise ValueError(f"Unsupported immunization profile '{profile_name}'. Available profiles: {available}")
    return IMMUNIZATION_PROFILES[profile_name]


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


def encode_prompt_variant(pipeline, prompt, device, do_classifier_free_guidance):
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        1,
        do_classifier_free_guidance,
        negative_prompt="",
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds


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


def build_boundary_ring(mask_tensor, ring_width):
    if ring_width <= 0:
        return torch.zeros_like(mask_tensor)

    kernel_size = ring_width * 2 + 1
    dilated = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=ring_width)
    eroded = 1 - F.max_pool2d(1 - mask_tensor, kernel_size=kernel_size, stride=1, padding=ring_width)
    return (dilated - eroded).clamp(0, 1)


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
    prompt_embeds,
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
    noise_pred = pipeline.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def apply_random_view(X_adv, base_image, perturbation_mask, config):
    if config.resize_jitter <= 0 and config.noise_strength <= 0 and config.blur_kernel_size <= 1:
        return X_adv

    height, width = X_adv.shape[-2:]
    min_scale = max(0.5, 1.0 - config.resize_jitter)
    max_scale = 1.0 + config.resize_jitter
    scale = float(torch.empty(1, device=X_adv.device).uniform_(min_scale, max_scale).item())
    resized_h = max(32, int(round(height * scale / 32) * 32))
    resized_w = max(32, int(round(width * scale / 32) * 32))

    transformed = F.interpolate(X_adv, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    transformed = F.interpolate(transformed, size=(height, width), mode="bilinear", align_corners=False)

    if config.blur_kernel_size > 1 and torch.rand(1, device=X_adv.device).item() < 0.5:
        kernel_size = config.blur_kernel_size
        transformed = F.avg_pool2d(
            transformed,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    if config.noise_strength > 0:
        noise = torch.randn_like(transformed) * config.noise_strength
        transformed = transformed + noise * perturbation_mask

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

        total_losses = []
        metric_values = {}
        for _ in range(max(1, config.eot_samples)):
            transformed = apply_random_view(X_adv, X, mask, config)
            total_loss, metrics = profile.compute_loss(
                transformed,
                reference_data,
                config,
                pipeline,
                transformed.dtype,
            )
            total_losses.append(total_loss)
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)
        loss = torch.stack(total_losses).mean()
        metric_summary = " | ".join(
            f"{name} {torch.stack(values).mean().item():.3f}"
            for name, values in metric_values.items()
        )
        pbar.set_description(f"Loss {loss.item():.5f} | {metric_summary} | step {actual_step_size:.4}")

        if progress_callback is not None:
            summarized_metrics = {
                name: float(torch.stack(values).mean().item())
                for name, values in metric_values.items()
            }
            progress_callback(
                {
                    "iteration": i + 1,
                    "totalIterations": iters,
                    "loss": float(loss.item()),
                    "stepSize": float(actual_step_size),
                    "metrics": summarized_metrics,
                }
            )

        grad, = torch.autograd.grad(loss, [X_adv])

        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv = X_adv.clamp(min=clamp_min, max=clamp_max).detach()

        if mask is not None:
            X_adv = X + (X_adv - X) * mask

    return X_adv


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

    try:
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            adv_X = pgd_optimize(
                X,
                profile=profile,
                reference_data=reference_data,
                pipeline=pipeline,
                config=config,
                eps=config.eps,
                step_size=config.step_size,
                iters=config.iters,
                clamp_min=config.clamp_min,
                clamp_max=config.clamp_max,
                mask=perturbation_mask,
                progress_callback=progress_callback,
            )
        adv_X = profile.finalize_adversarial(adv_X, X, perturbation_mask, config)
    finally:
        try:
            del X, mask, masked_image
        except Exception:
            pass

    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    adv_image = topil(adv_X[0].detach().cpu()).convert("RGB")
    adv_image = recover_image(adv_image, init_image, mask_image, background=True)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return adv_image, reference_data.get("target_image")
