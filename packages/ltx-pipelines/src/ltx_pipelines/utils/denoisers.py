"""Flat denoiser classes — transformer received at call time, not stored.
Three implementations of the :class:`~ltx_pipelines.utils.types.Denoiser` protocol:
* :class:`SimpleDenoiser` — single transformer call, no guidance.
* :class:`GuidedDenoiser` — static guiders, handles CFG + STG + isolated modality.
* :class:`FactoryGuidedDenoiser` — resolves guiders per-step from sigma.
``GuidedDenoiser`` and ``FactoryGuidedDenoiser`` share the core multi-pass
logic via the module-level :func:`_guided_denoise` function, which runs each
guidance pass as its own transformer forward (batch ``B``), then combines
outputs via the guiders. Pass outputs are stashed on **CPU** after each forward
so multiple CFG/STG/modality passes do not all occupy GPU VRAM at once; tensors
are moved back to each latent's device only for ``guider.calculate``.
"""

import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state

_POSITIVE_ONLY_GUIDER = MultiModalGuider(
    params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0),
)
"""Guider that only runs the conditioned pass and returns cond unchanged."""


def _ensure_guider(guider: MultiModalGuider | None) -> MultiModalGuider:
    """Return the guider as-is, or a positive-only guider for absent modalities."""
    return guider if guider is not None else _POSITIVE_ONLY_GUIDER


def _pass_output_to_cpu(t: torch.Tensor | float | None) -> torch.Tensor | float | None:
    """Drop GPU references for a transformer output (keep a detached CPU copy)."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return t


def _to_guider_compute_device(t: torch.Tensor | float | None, device: torch.device | None) -> torch.Tensor | float | None:
    """Move tensor to *device* for guider fusion; scalars and ``None`` unchanged."""
    if device is None or not isinstance(t, torch.Tensor):
        return t
    return t.to(device=device, non_blocking=True)


def _guided_denoise(  # noqa: PLR0913
    transformer: X0Model,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    sigma: torch.Tensor,
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor | None,
    a_context: torch.Tensor | None,
    *,
    last_denoised_video: torch.Tensor | None,
    last_denoised_audio: torch.Tensor | None,
    step_index: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Core guided denoising — one transformer forward per guidance pass.

    Each pass uses the original latent batch size ``B`` (typically 1) with that
    pass's text/audio context, instead of stacking ``n`` passes into one forward
    with batch ``n*B``. Intermediate outputs live on CPU until
    ``guider.calculate`` so peak VRAM does not scale with the number of passes.
    Guiders must not be ``None``. For absent modalities, callers should pass
    :data:`_POSITIVE_ONLY_GUIDER` (via :func:`_ensure_guider`) so that only
    the conditioned pass runs and ``calculate()`` returns cond unchanged.
    """
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        return last_denoised_video, last_denoised_audio

    if video_state is not None and v_context is None:
        raise ValueError("v_context is required when video_state is provided")
    if audio_state is not None and a_context is None:
        raise ValueError("a_context is required when audio_state is provided")
    _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
    passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

    if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
        if video_guider.do_unconditional_generation() and video_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        if audio_guider.do_unconditional_generation() and audio_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations: list[Perturbation] = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
        )
    if stg_perturbations:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                        Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                    ]
                ),
            )
        )

    results: dict[str, tuple[torch.Tensor | float | None, torch.Tensor | float | None]] = {}
    for name, vc, ac, ptb in passes:
        bv = None
        ba = None
        if video_state is not None and vc is not None:
            dev_v = video_state.latent.device
            sig_v = sigma.expand(video_state.latent.shape[0])
            bv = modality_from_latent_state(
                video_state,
                vc.to(dev_v, non_blocking=True),
                sig_v,
                enabled=not v_skip,
            )
        if audio_state is not None and ac is not None:
            dev_a = audio_state.latent.device
            sig_a = sigma.expand(audio_state.latent.shape[0])
            ba = modality_from_latent_state(
                audio_state,
                ac.to(dev_a, non_blocking=True),
                sig_a,
                enabled=not a_skip,
            )
        dv, da = transformer(
            video=bv,
            audio=ba,
            perturbations=BatchedPerturbationConfig([ptb]),
        )
        results[name] = (_pass_output_to_cpu(dv), _pass_output_to_cpu(da))
        del dv, da

    v_dev = video_state.latent.device if video_state is not None else None
    a_dev = audio_state.latent.device if audio_state is not None else None

    cond_v, cond_a = results["cond"]
    uncond_v, uncond_a = results.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = results.get("ptb", (0.0, 0.0))
    mod_v, mod_a = results.get("mod", (0.0, 0.0))

    cond_v = _to_guider_compute_device(cond_v, v_dev)
    uncond_v = _to_guider_compute_device(uncond_v, v_dev)
    ptb_v = _to_guider_compute_device(ptb_v, v_dev)
    mod_v = _to_guider_compute_device(mod_v, v_dev)

    cond_a = _to_guider_compute_device(cond_a, a_dev)
    uncond_a = _to_guider_compute_device(uncond_a, a_dev)
    ptb_a = _to_guider_compute_device(ptb_a, a_dev)
    mod_a = _to_guider_compute_device(mod_a, a_dev)

    denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
    denoised_audio = last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
    return denoised_video, denoised_audio


class SimpleDenoiser:
    """Single transformer call, no guidance.
    Passes ``None`` Modality for absent modalities.
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        return transformer(video=pos_video, audio=pos_audio, perturbations=None)


class GuidedDenoiser:
    """Static guiders — handles CFG + STG + isolated modality.
    Context/guider can be ``None`` for absent modalities (a positive-only
    guider is substituted at call time).
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider: MultiModalGuider | None = None,
        audio_guider: MultiModalGuider | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        denoised_video, denoised_audio = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=_ensure_guider(self.video_guider),
            audio_guider=_ensure_guider(self.audio_guider),
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
        )
        self._last_denoised_video = denoised_video
        self._last_denoised_audio = denoised_audio
        return denoised_video, denoised_audio


class FactoryGuidedDenoiser:
    """Resolves guiders per-step from sigma, then delegates to shared guided logic."""

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider_factory: MultiModalGuiderFactory | None = None,
        audio_guider_factory: MultiModalGuiderFactory | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider_factory = video_guider_factory
        self.audio_guider_factory = audio_guider_factory
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None
        self._sigma_vals_cached: list[float] | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._sigma_vals_cached is None:
            self._sigma_vals_cached = sigmas.detach().cpu().tolist()
        sigma_val = self._sigma_vals_cached[step_index]

        video_guider = _ensure_guider(
            self.video_guider_factory.build_from_sigma(sigma_val) if self.video_guider_factory else None
        )
        audio_guider = _ensure_guider(
            (self.audio_guider_factory or self.video_guider_factory).build_from_sigma(sigma_val)
            if self.video_guider_factory or self.audio_guider_factory
            else None
        )

        denoised_video, denoised_audio = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=video_guider,
            audio_guider=audio_guider,
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
        )
        self._last_denoised_video = denoised_video
        self._last_denoised_audio = denoised_audio
        return denoised_video, denoised_audio
