import type { ImmunizationProfile } from "../types/api";

export interface ImmunizationProfileDetail {
  label: string;
  summary: string;
}

export interface ImmunizationProfileGroup {
  label: string;
  options: Array<{
    value: ImmunizationProfile;
    label: string;
  }>;
}

export const IMMUNIZATION_PROFILE_DETAILS: Record<ImmunizationProfile, ImmunizationProfileDetail> = {
  stable_diffusion: {
    label: "Stable Diffusion layered",
    summary:
      "Upgraded SD proxy with boundary, descriptor, and context-anchor losses. Good first pass for DeeVid-style latent edits.",
  },
  full_regeneration_scaffold: {
    label: "Scaffold: full regeneration",
    summary:
      "Research scaffold aimed at whole-image redraw attacks. Useful for probing regeneration-heavy editors, not the main production profile.",
  },
  instruction_editing_scaffold: {
    label: "Scaffold: instruction editing",
    summary:
      "Research scaffold for instruction-guided edits that change a single attribute or object while keeping the rest realistic.",
  },
  controlnet_scaffold: {
    label: "Scaffold: ControlNet / structure lock",
    summary:
      "Research scaffold for pose- and geometry-preserving pipelines where structure stays fixed but texture or identity changes.",
  },
  style_transfer_scaffold: {
    label: "Scaffold: style transfer",
    summary:
      "Research scaffold for aesthetic restyling and look-transfer pipelines that preserve composition while rewriting texture.",
  },
  artist_cloak: {
    label: "Artist cloak",
    summary:
      "Artwork-focused profile with Glaze/Mist-style texture cloaking plus a stronger ArtShield-like ownership watermark path.",
  },
  text_aware_scaffold: {
    label: "Scaffold: text-aware",
    summary:
      "Research scaffold for editors that clean up overlays, logos, and nearby text while keeping the rest of the scene stable.",
  },
  adversarial_hardened_scaffold: {
    label: "Scaffold: adversarial hardened",
    summary:
      "Research scaffold for stronger or fine-tuned editors that try to preserve visible quality while bypassing defenses.",
  },
  nano_banana_experimental: {
    label: "Nano Banana experimental",
    summary:
      "Earlier layered Nano Banana profile. Lighter and faster than the newer NB2 stack, but generally less robust.",
  },
  nano_banana_2: {
    label: "Nano Banana 2 layered (recommended)",
    summary:
      "Best balanced upgraded profile for OpenArt-style reference, cleanup, and character-consistency edits. Recommended starting point.",
  },
  nano_banana_2_hard_block: {
    label: "Nano Banana 2 hard block",
    summary:
      "Stricter fail-closed profile with stronger tripwires. More likely to refuse on tight GPUs instead of silently weakening the defense.",
  },
  nano_banana_2_distortion: {
    label: "Nano Banana 2 distortion block",
    summary:
      "Strongest and most visually aggressive profile. Best reserved for hostile edits when visible distortion is acceptable.",
  },
};

export const IMMUNIZATION_PROFILE_GROUPS: ImmunizationProfileGroup[] = [
  {
    label: "Recommended",
    options: [
      { value: "nano_banana_2", label: IMMUNIZATION_PROFILE_DETAILS.nano_banana_2.label },
      { value: "stable_diffusion", label: IMMUNIZATION_PROFILE_DETAILS.stable_diffusion.label },
    ],
  },
  {
    label: "Strict",
    options: [
      { value: "nano_banana_2_hard_block", label: IMMUNIZATION_PROFILE_DETAILS.nano_banana_2_hard_block.label },
      { value: "nano_banana_2_distortion", label: IMMUNIZATION_PROFILE_DETAILS.nano_banana_2_distortion.label },
    ],
  },
  {
    label: "Research",
    options: [
      { value: "nano_banana_experimental", label: IMMUNIZATION_PROFILE_DETAILS.nano_banana_experimental.label },
      { value: "full_regeneration_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.full_regeneration_scaffold.label },
      { value: "instruction_editing_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.instruction_editing_scaffold.label },
      { value: "controlnet_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.controlnet_scaffold.label },
      { value: "style_transfer_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.style_transfer_scaffold.label },
      { value: "artist_cloak", label: IMMUNIZATION_PROFILE_DETAILS.artist_cloak.label },
      { value: "text_aware_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.text_aware_scaffold.label },
      { value: "adversarial_hardened_scaffold", label: IMMUNIZATION_PROFILE_DETAILS.adversarial_hardened_scaffold.label },
    ],
  },
];

export function profileLabel(profile: ImmunizationProfile | null) {
  return profile ? IMMUNIZATION_PROFILE_DETAILS[profile].label : null;
}

export function profileSummary(profile: ImmunizationProfile | null) {
  return profile ? IMMUNIZATION_PROFILE_DETAILS[profile].summary : null;
}
