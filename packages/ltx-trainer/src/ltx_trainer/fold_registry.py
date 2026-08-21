"""Registry for folding paper-stub modules into LTX preprocess / save paths.

Enable hooks via ``GOPEX_AV_FOLD_HOOKS`` (comma-separated), e.g.::

    export GOPEX_AV_FOLD_HOOKS=fmelcodec,speech_quality_emb,robustspeechflow

Hooks run at **save time** and attach JSON-serializable metadata (and optional tensors)
to ``.pt`` sidecars without changing core LTX tensor shapes unless noted.
"""

from __future__ import annotations

import os
from typing import Any, Callable

AnnotateFn = Callable[[dict[str, Any]], dict[str, Any]]

_HOOKS: dict[str, AnnotateFn] = {}


def _register(name: str, fn: AnnotateFn) -> None:
    _HOOKS[name] = fn


def enabled_hook_names() -> list[str]:
    """Active hooks for encode/save. Honors ``GOPEX_AV_FOLD_HOOKS`` or ``GOPEX_ENABLE_AV_FOLD=1``."""
    raw = os.environ.get("GOPEX_AV_FOLD_HOOKS", "").strip()
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    if os.environ.get("GOPEX_ENABLE_AV_FOLD", "").strip().lower() in ("1", "true", "yes"):
        return fold_hooks_for_preprocess_meta()
    return []


def list_registered_hooks() -> list[str]:
    _ensure_hooks_loaded()
    return sorted(_HOOKS.keys())


def _ensure_hooks_loaded() -> None:
    if _HOOKS:
        return
    from ltx_trainer.adamag import fold as adamag_fold
    from ltx_trainer.ag_repa import fold as ag_repa_fold
    from ltx_trainer.avbench import fold as avbench_fold
    from ltx_trainer.mtavg2 import fold as mtavg2_fold
    from ltx_trainer.cert_las import fold as cert_las_fold
    from ltx_trainer.core_kd import fold as core_kd_fold
    from ltx_trainer.flatsounds import fold as flatsounds_fold
    from ltx_trainer.vlm_count import fold as vlm_count_fold
    from ltx_trainer.pesd_vit import fold as pesd_vit_fold
    from ltx_trainer.entroad import fold as entroad_fold
    from ltx_trainer.eigenet import fold as eigenet_fold
    from ltx_trainer.planaudio import fold as planaudio_fold
    from ltx_trainer.bernini import fold as bernini_fold
    from ltx_trainer.fmelcodec import fold as fmelcodec_fold
    from ltx_trainer.holitok import fold as holitok_fold
    from ltx_trainer.comet import fold as comet_fold
    from ltx_trainer.cafnet import fold as cafnet_fold
    from ltx_trainer.childvox import fold as childvox_fold
    from ltx_trainer.btscafe import fold as btscafe_fold
    from ltx_trainer.dlmasr import fold as dlmasr_fold
    from ltx_trainer.cryacc import fold as cryacc_fold
    from ltx_trainer.demon import fold as demon_fold
    from ltx_trainer.dasheng_audiogen import fold as dasheng_audiogen_fold
    from ltx_trainer.voicegiraffe import fold as voicegiraffe_fold
    from ltx_trainer.growloop import fold as growloop_fold
    from ltx_trainer.mustbench import fold as mustbench_fold
    from ltx_trainer.lamo import fold as lamo_fold
    from ltx_trainer.longav_compass import fold as longav_fold
    from ltx_trainer.autocut import fold as autocut_fold
    from ltx_trainer.omnicustom import fold as omnicustom_fold
    from ltx_trainer.id_lora import fold as id_lora_fold
    from ltx_trainer.phyworld import fold as phyworld_fold
    from ltx_trainer.robustspeechflow import fold as rsf_fold
    from ltx_trainer.speech_quality_emb import fold as sqe_fold
    from ltx_trainer.lumivid import fold as lumivid_fold
    from ltx_trainer.modulo_spike_hdr import fold as modulo_spike_fold
    from ltx_trainer.physthdr_gs import fold as physthdr_fold
    from ltx_trainer.x2hdr import fold as x2hdr_fold
    from ltx_trainer.lf_diff import fold as lf_diff_fold
    from ltx_trainer.raim_mef import fold as raim_mef_fold
    from ltx_trainer.stem2_sdr_hdr import fold as stem2_fold
    from ltx_trainer.latenthdr import fold as latenthdr_fold
    from ltx_trainer.lucky_hdr import fold as lucky_hdr_fold
    from ltx_trainer.sdr2hdr import fold as sdr2hdr_fold
    from ltx_trainer.diffhdr import fold as diffhdr_fold
    from ltx_trainer.vdp_hdr import fold as vdp_hdr_fold
    from ltx_trainer.cubediff import fold as cubediff_fold
    from ltx_trainer.spherediff import fold as spherediff_fold
    from ltx_trainer.spherefusion import fold as spherefusion_fold
    from ltx_trainer.sphereuformer import fold as sphereuformer_fold
    from ltx_trainer.sphere_depth import fold as sphere_depth_fold
    from ltx_trainer.pantheon360 import fold as pantheon360_fold
    from ltx_trainer.semantic_stitch import fold as semantic_stitch_fold
    from ltx_trainer.nvc_erp_qpa import fold as nvc_erp_qpa_fold
    from ltx_trainer.dense360 import fold as dense360_fold
    from ltx_trainer.sphere360 import fold as sphere360_fold
    from ltx_trainer.pano360 import fold as pano360_fold
    from ltx_trainer.anything360 import fold as anything360_fold
    from ltx_trainer.cross360 import fold as cross360_fold
    from ltx_trainer.erpgs import fold as erpgs_fold
    from ltx_trainer.panoworld import fold as panoworld_fold
    from ltx_trainer.panoenv import fold as panoenv_fold
    from ltx_trainer.gimbal360 import fold as gimbal360_fold
    from ltx_trainer.mtpano import fold as mtpano_fold
    from ltx_trainer.panoworld_x import fold as panoworld_x_fold
    from ltx_trainer.pano_affordance import fold as pano_affordance_fold
    from ltx_trainer.panogsdet import fold as panogsdet_fold
    from ltx_trainer.panolm import fold as panolm_fold
    from ltx_trainer.pano_flight import fold as pano_flight_fold
    from ltx_trainer.humanview_vu import fold as humanview_vu_fold
    from ltx_trainer.clairvoyant import fold as clairvoyant_fold
    from ltx_trainer.ai_hpc_workflows import fold as ai_hpc_workflows_fold
    from ltx_trainer.pccl import fold as pccl_fold
    from ltx_trainer.fdtd_cpml_multigpu import fold as fdtd_cpml_multigpu_fold
    from ltx_trainer.midint_division import fold as midint_division_fold
    from ltx_trainer.deopt_reopt import fold as deopt_reopt_fold
    from ltx_trainer.nvshmem_demystify import fold as nvshmem_demystify_fold
    from ltx_trainer.latent_prm_guidance import fold as latent_prm_guidance_fold
    from ltx_trainer.set_cuda_graph import fold as set_cuda_graph_fold
    from ltx_trainer.radiusfps import fold as radiusfps_fold
    from ltx_trainer.s3u_sar import fold as s3u_sar_fold
    from ltx_trainer.era_defocus import fold as era_defocus_fold
    from ltx_trainer.flood_physics import fold as flood_physics_fold
    from ltx_trainer.dilated_sym_diff import fold as dilated_sym_diff_fold
    from ltx_trainer.dynamic_gp import fold as dynamic_gp_fold
    from ltx_trainer.dyna_pruner import fold as dyna_pruner_fold
    from ltx_trainer.branch_energy import fold as branch_energy_fold
    from ltx_trainer.lora_hd_attn import fold as lora_hd_attn_fold
    from ltx_trainer.fp8_ozaki import fold as fp8_ozaki_fold
    from ltx_trainer.predictive_autoscaling import fold as predictive_autoscaling_fold
    from ltx_trainer.weatherproof import fold as weatherproof_fold
    from ltx_trainer.s3po import fold as s3po_fold
    from ltx_trainer.vsr_vqa import fold as vsr_fold
    from ltx_trainer.vuga import fold as vuga_fold
    from ltx_trainer.vuboiqa import fold as vuboiqa_fold
    from ltx_trainer.wavenext2 import fold as wn2_fold
    from ltx_trainer.swansphere import fold as swansphere_fold
    from ltx_trainer.foley_omni import fold as foley_omni_fold
    from ltx_trainer.fuse_flow import fold as fuse_flow_fold
    from ltx_trainer.forte import fold as forte_fold
    from ltx_trainer.svhighlights import fold as svhighlights_fold
    from ltx_trainer.evogs import fold as evogs_fold
    from ltx_trainer.mmae import fold as mmae_fold
    from ltx_trainer.mirage import fold as mirage_fold

    _register("eigenet", eigenet_fold.annotate_audio_save_data)
    _register("ag_repa", ag_repa_fold.annotate_audio_save_data)
    _register("swansphere_audio", swansphere_fold.annotate_audio_save_data)
    _register("foley_omni_audio", foley_omni_fold.annotate_audio_save_data)
    _register("planaudio", planaudio_fold.annotate_audio_save_data)
    _register("fmelcodec", fmelcodec_fold.annotate_audio_save_data)
    _register("holitok", holitok_fold.annotate_audio_save_data)
    _register("comet", comet_fold.annotate_audio_save_data)
    _register("cafnet", cafnet_fold.annotate_audio_save_data)
    _register("childvox", childvox_fold.annotate_audio_save_data)
    _register("btscafe", btscafe_fold.annotate_audio_save_data)
    _register("dlmasr", dlmasr_fold.annotate_audio_save_data)
    _register("cryacc", cryacc_fold.annotate_audio_save_data)
    _register("demon", demon_fold.annotate_audio_save_data)
    _register("voicegiraffe", voicegiraffe_fold.annotate_audio_save_data)
    _register("dasheng_audiogen", dasheng_audiogen_fold.annotate_audio_save_data)
    _register("growloop", growloop_fold.annotate_audio_save_data)
    _register("mustbench", mustbench_fold.annotate_audio_save_data)
    _register("speech_quality_emb", sqe_fold.annotate_audio_save_data)
    _register("robustspeechflow", rsf_fold.annotate_audio_save_data)
    _register("wavenext2", wn2_fold.annotate_audio_save_data)
    _register("adamag", adamag_fold.annotate_video_latent_data)
    _register("vsr_vqa", vsr_fold.annotate_video_latent_data)
    _register("vuga", vuga_fold.annotate_video_latent_data)
    _register("vuboiqa", vuboiqa_fold.annotate_video_latent_data)
    _register("s3po", s3po_fold.annotate_video_latent_data)
    _register("modulo_spike_hdr", modulo_spike_fold.annotate_video_latent_data)
    _register("physthdr_gs", physthdr_fold.annotate_video_latent_data)
    _register("lumivid", lumivid_fold.annotate_video_latent_data)
    _register("x2hdr", x2hdr_fold.annotate_video_latent_data)
    _register("lf_diff", lf_diff_fold.annotate_video_latent_data)
    _register("raim_mef", raim_mef_fold.annotate_video_latent_data)
    _register("stem2_sdr_hdr", stem2_fold.annotate_video_latent_data)
    _register("latenthdr", latenthdr_fold.annotate_video_latent_data)
    _register("lucky_hdr", lucky_hdr_fold.annotate_video_latent_data)
    _register("sdr2hdr", sdr2hdr_fold.annotate_video_latent_data)
    _register("diffhdr", diffhdr_fold.annotate_video_latent_data)
    _register("vdp_hdr", vdp_hdr_fold.annotate_video_latent_data)
    _register("cubediff", cubediff_fold.annotate_video_latent_data)
    _register("spherediff", spherediff_fold.annotate_video_latent_data)
    _register("spherefusion", spherefusion_fold.annotate_video_latent_data)
    _register("sphereuformer", sphereuformer_fold.annotate_video_latent_data)
    _register("sphere_depth", sphere_depth_fold.annotate_video_latent_data)
    _register("pantheon360", pantheon360_fold.annotate_video_latent_data)
    _register("semantic_stitch", semantic_stitch_fold.annotate_video_latent_data)
    _register("nvc_erp_qpa", nvc_erp_qpa_fold.annotate_video_latent_data)
    _register("dense360", dense360_fold.annotate_video_latent_data)
    _register("sphere360", sphere360_fold.annotate_video_latent_data)
    _register("pano360", pano360_fold.annotate_video_latent_data)
    _register("anything360", anything360_fold.annotate_video_latent_data)
    _register("cross360", cross360_fold.annotate_video_latent_data)
    _register("erpgs", erpgs_fold.annotate_video_latent_data)
    _register("panoworld", panoworld_fold.annotate_video_latent_data)
    _register("panoenv", panoenv_fold.annotate_video_latent_data)
    _register("gimbal360", gimbal360_fold.annotate_video_latent_data)
    _register("mtpano", mtpano_fold.annotate_video_latent_data)
    _register("panoworld_x", panoworld_x_fold.annotate_video_latent_data)
    _register("pano_affordance", pano_affordance_fold.annotate_video_latent_data)
    _register("panogsdet", panogsdet_fold.annotate_video_latent_data)
    _register("panolm", panolm_fold.annotate_video_latent_data)
    _register("pano_flight", pano_flight_fold.annotate_video_latent_data)
    _register("humanview_vu", humanview_vu_fold.annotate_video_latent_data)
    _register("clairvoyant", clairvoyant_fold.annotate_video_latent_data)
    _register("ai_hpc_workflows", ai_hpc_workflows_fold.annotate_video_latent_data)
    _register("pccl", pccl_fold.annotate_video_latent_data)
    _register("fdtd_cpml_multigpu", fdtd_cpml_multigpu_fold.annotate_video_latent_data)
    _register("midint_division", midint_division_fold.annotate_video_latent_data)
    _register("deopt_reopt", deopt_reopt_fold.annotate_video_latent_data)
    _register("nvshmem_demystify", nvshmem_demystify_fold.annotate_video_latent_data)
    _register("latent_prm_guidance", latent_prm_guidance_fold.annotate_video_latent_data)
    _register("set_cuda_graph", set_cuda_graph_fold.annotate_video_latent_data)
    _register("radiusfps", radiusfps_fold.annotate_video_latent_data)
    _register("s3u_sar", s3u_sar_fold.annotate_video_latent_data)
    _register("era_defocus", era_defocus_fold.annotate_video_latent_data)
    _register("flood_physics", flood_physics_fold.annotate_video_latent_data)
    _register("dilated_sym_diff", dilated_sym_diff_fold.annotate_video_latent_data)
    _register("dynamic_gp", dynamic_gp_fold.annotate_video_latent_data)
    _register("dyna_pruner", dyna_pruner_fold.annotate_video_latent_data)
    _register("branch_energy", branch_energy_fold.annotate_video_latent_data)
    _register("lora_hd_attn", lora_hd_attn_fold.annotate_video_latent_data)
    _register("fp8_ozaki", fp8_ozaki_fold.annotate_video_latent_data)
    _register("predictive_autoscaling", predictive_autoscaling_fold.annotate_video_latent_data)
    _register("weatherproof", weatherproof_fold.annotate_video_latent_data)
    _register("lamo", lamo_fold.annotate_video_latent_data)
    _register("bernini", bernini_fold.annotate_video_latent_data)
    _register("phyworld", phyworld_fold.annotate_video_latent_data)
    _register("mirage", mirage_fold.annotate_video_latent_data)
    _register("fuse_flow", fuse_flow_fold.annotate_video_latent_data)
    _register("forte", forte_fold.annotate_video_latent_data)
    _register("forte_audio", forte_fold.annotate_audio_save_data)
    _register("svhighlights", svhighlights_fold.annotate_video_latent_data)
    _register("evogs", evogs_fold.annotate_video_latent_data)
    _register("mmae", mmae_fold.annotate_video_latent_data)
    _register("mmae_audio", mmae_fold.annotate_audio_save_data)
    _register("avbench", avbench_fold.annotate_video_latent_data)
    _register("cert_las", cert_las_fold.annotate_video_latent_data)
    _register("flatsounds", flatsounds_fold.annotate_video_latent_data)
    _register("core_kd", core_kd_fold.annotate_video_latent_data)
    _register("vlm_count", vlm_count_fold.annotate_video_latent_data)
    _register("pesd_vit", pesd_vit_fold.annotate_video_latent_data)
    _register("entroad", entroad_fold.annotate_video_latent_data)
    _register("mtavg2", mtavg2_fold.annotate_video_latent_data)
    _register("longav_compass", longav_fold.annotate_video_latent_data)
    _register("autocut", autocut_fold.annotate_video_latent_data)
    _register("autocut_audio", autocut_fold.annotate_audio_save_data)
    _register("omnicustom", omnicustom_fold.annotate_video_latent_data)
    _register("omnicustom_audio", omnicustom_fold.annotate_audio_save_data)
    _register("id_lora", id_lora_fold.annotate_video_latent_data)
    _register("id_lora_audio", id_lora_fold.annotate_audio_save_data)


def fold_hooks_for_preprocess_meta() -> list[str]:
    """Hooks that would run when ``GOPEX_ENABLE_AV_FOLD=1`` (for preprocess_meta.json)."""
    raw = os.environ.get("GOPEX_AV_FOLD_HOOKS", "").strip()
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [
        "vsr_vqa",
        "vuga",
        "vuboiqa",
        "s3po",
        "lucky_hdr",
        "adamag",
        "lamo",
        "bernini",
        "phyworld",
        "mirage",
        "fuse_flow",
        "forte",
        "forte_audio",
        "svhighlights",
        "evogs",
        "eigenet",
        "planaudio",
        "fmelcodec",
        "holitok",
        "comet",
        "cafnet",
        "childvox",
        "btscafe",
        "dlmasr",
        "cryacc",
        "demon",
        "voicegiraffe",
        "dasheng_audiogen",
        "growloop",
        "mustbench",
        "speech_quality_emb",
        "robustspeechflow",
        "wavenext2",
        "avbench",
        "entroad",
        "cert_las",
        "flatsounds",
        "core_kd",
        "vlm_count",
        "mtavg2",
        "pesd_vit",
        "longav_compass",
        "autocut",
        "autocut_audio",
        "omnicustom",
        "omnicustom_audio",
        "id_lora",
        "id_lora_audio",
        "ag_repa",
        "swansphere_audio",
        "foley_omni_audio",
        "modulo_spike_hdr",
        "physthdr_gs",
        "lumivid",
        "x2hdr",
        "lf_diff",
        "raim_mef",
        "stem2_sdr_hdr",
        "latenthdr",
        "sdr2hdr",
        "diffhdr",
        "vdp_hdr",
        "cubediff",
        "spherediff",
        "spherefusion",
        "sphereuformer",
        "sphere_depth",
        "pantheon360",
        "semantic_stitch",
        "nvc_erp_qpa",
        "dense360",
        "sphere360",
        "pano360",
        "anything360",
        "cross360",
        "erpgs",
        "panoworld",
        "panoenv",
        "gimbal360",
        "mtpano",
        "panoworld_x",
        "pano_affordance",
        "panogsdet",
        "panolm",
        "pano_flight",
        "humanview_vu",
        "clairvoyant",
        "predictive_autoscaling",
        "ai_hpc_workflows",
        "pccl",
        "fdtd_cpml_multigpu",
        "midint_division",
        "deopt_reopt",
        "nvshmem_demystify",
        "latent_prm_guidance",
        "set_cuda_graph",
        "radiusfps",
        "s3u_sar",
        "era_defocus",
        "flood_physics",
        "dilated_sym_diff",
        "dynamic_gp",
        "branch_energy",
        "lora_hd_attn",
        "weatherproof",
    ]


_VIDEO_HOOKS = (
    "adamag",
    "vsr_vqa",
    "vuga",
    "vuboiqa",
    "s3po",
    "modulo_spike_hdr",
    "physthdr_gs",
    "lumivid",
    "x2hdr",
    "lf_diff",
    "raim_mef",
    "stem2_sdr_hdr",
    "latenthdr",
    "lucky_hdr",
    "sdr2hdr",
    "diffhdr",
    "vdp_hdr",
    "cubediff",
    "spherediff",
    "spherefusion",
    "sphereuformer",
    "sphere_depth",
    "pantheon360",
    "semantic_stitch",
    "nvc_erp_qpa",
    "dense360",
    "sphere360",
    "pano360",
    "anything360",
    "cross360",
    "erpgs",
    "panoworld",
    "panoenv",
    "gimbal360",
    "mtpano",
    "panoworld_x",
    "pano_affordance",
    "panogsdet",
    "panolm",
    "pano_flight",
    "humanview_vu",
    "clairvoyant",
    "predictive_autoscaling",
    "weatherproof",
    "lamo",
    "bernini",
    "phyworld",
    "mirage",
    "fuse_flow",
    "forte",
    "avbench",
    "entroad",
    "cert_las",
    "flatsounds",
    "core_kd",
    "vlm_count",
    "pesd_vit",
    "mtavg2",
    "longav_compass",
    "autocut",
    "omnicustom",
    "id_lora",
    "mmae",
    "era_defocus",
    "flood_physics",
    "dilated_sym_diff",
    "dynamic_gp",
    "branch_energy",
    "lora_hd_attn",
)


_AUDIO_HOOKS = (
    "eigenet",
    "planaudio",
    "fmelcodec",
    "holitok",
    "comet",
    "cafnet",
    "childvox",
    "btscafe",
    "dlmasr",
    "cryacc",
    "demon",
    "voicegiraffe",
    "dasheng_audiogen",
    "growloop",
    "mustbench",
    "speech_quality_emb",
    "robustspeechflow",
    "wavenext2",
    "autocut_audio",
    "omnicustom_audio",
    "id_lora_audio",
    "mmae_audio",
    "ag_repa",
    "forte_audio",
    "swansphere_audio",
    "foley_omni_audio",
)


def expected_video_fold_keys(hooks: list[str] | None = None) -> tuple[str, ...]:
    """Sidecar keys expected on video ``latents/*.pt`` for the given hook list."""
    active = hooks if hooks is not None else enabled_hook_names()
    video = frozenset(_VIDEO_HOOKS)
    return tuple(h for h in active if h in video)


def expected_audio_fold_keys(hooks: list[str] | None = None) -> tuple[str, ...]:
    """Sidecar keys expected on ``audio_latents/*.pt`` for the given hook list."""
    active = hooks if hooks is not None else enabled_hook_names()
    audio = frozenset(_AUDIO_HOOKS)
    return tuple(h for h in active if h in audio)


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Run enabled audio hooks in order; each may add a top-level metadata key."""
    names = enabled_hook_names()
    if not names:
        return data
    _ensure_hooks_loaded()
    video_only = frozenset(_VIDEO_HOOKS)
    out = dict(data)
    for name in names:
        if name in video_only:
            continue
        fn = _HOOKS.get(name)
        if fn is not None:
            out = fn(out)
    return out


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Run enabled video-side hooks (quality, guidance, motion, planning, physics)."""
    names = enabled_hook_names()
    if not names:
        return data
    _ensure_hooks_loaded()
    out = dict(data)
    for name in names:
        if name in _VIDEO_HOOKS:
            fn = _HOOKS.get(name)
            if fn is not None:
                out = fn(out)
    return out


def fold_status() -> dict[str, Any]:
    """Summary for CLI / audit."""
    _ensure_hooks_loaded()
    return {
        "enabled": enabled_hook_names(),
        "registered": list_registered_hooks(),
    }
