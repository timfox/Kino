"""OmniCustom evaluation smoke (arXiv:2602.12304)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.omnicustom.benchmark import BENCHMARK_CASES
from ltx_trainer.omnicustom.config import OmniCustomConfig
from ltx_trainer.omnicustom.dataset import dataset_card
from ltx_trainer.omnicustom.loss import total_omnicustom_loss
from ltx_trainer.omnicustom.metrics import stub_identity_metrics, stub_speaker_sim
from ltx_trainer.omnicustom.prompts import extract_speech
from ltx_trainer.omnicustom.reference_lora import inject_embedding, reference_self_attention
from ltx_trainer.omnicustom.tables import appendix_lse_comparison, table2_quantitative, table3_user_study
from ltx_trainer.omnicustom.taxonomy import CustomizationSetting, SETTING_TRAITS


def evaluation_smoke(cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    rng = np.random.default_rng(c.random_seed)

    # Reference LoRA attention stub
    d = 16
    q = rng.normal(size=(4, d))
    k = rng.normal(size=(4, d))
    v = rng.normal(size=(4, d))
    qr = rng.normal(size=(2, d))
    kr = rng.normal(size=(2, d))
    vr = rng.normal(size=(2, d))
    z, z_ref = reference_self_attention(q, k, v, qr, kr, vr)
    z = inject_embedding(z, rng.normal(size=(512,)))

    # Loss stub
    target = rng.normal(size=(8, d))
    pred = target + rng.normal(scale=0.05, size=target.shape)
    pred_no = target + rng.normal(scale=0.2, size=target.shape)
    loss = total_omnicustom_loss(
        v_pred=pred,
        v_target=target,
        a_pred=pred,
        a_target=target,
        v_pred_no_ref=pred_no,
        a_pred_no_ref=pred_no,
        cfg=c,
    )

    case = BENCHMARK_CASES[0]
    parsed = extract_speech(case.prompt)

    ref_face = rng.normal(size=(512,))
    frames = [ref_face + rng.normal(scale=0.05, size=512) for _ in range(5)]
    arc, cur = stub_identity_metrics(ref_face, frames)
    spk = stub_speaker_sim(rng.normal(size=256), rng.normal(size=256) * 0.9 + ref_face[:256] * 0.1)

    paper_row = [r for r in table2_quantitative() if r.get("model") == "OmniCustom (+ contrastive)"][0]

    return {
        "paper": f"arXiv:{c.paper_arxiv}",
        "project_url": c.project_url,
        "task": CustomizationSetting.SYNC_AV.value,
        "setting_traits": SETTING_TRAITS[CustomizationSetting.SYNC_AV],
        "base_model": c.base_model,
        "lora_rank": c.lora_rank,
        "loss": loss.to_dict(),
        "reference_attention_shapes": {"z": list(z.shape), "z_ref": list(z_ref.shape)},
        "benchmark_case": case.case_id,
        "parsed_speech": parsed.speech_text,
        "stub_facesim_arc": round(arc, 4),
        "stub_speaker_sim": round(spk, 4),
        "paper_facesim_arc": paper_row["facesim_arc"],
        "paper_fvd": paper_row["fvd"],
        "paper_speaker_sim": paper_row["speaker_sim"],
        "dataset": dataset_card(c),
        "table2_rows": len(table2_quantitative()),
        "table3_rows": len(table3_user_study()),
        "lse_best": appendix_lse_comparison()[-1],
    }
