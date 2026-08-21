"""Dual-output L2 ASR MTL entanglement — Cho & Kim, arXiv:2606.06065."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class L2DoEntangleConfig:
    paper_arxiv: str = "arXiv:2606.06065"
    title: str = (
        "Multi-task Learning is Not Enough: Representational Entanglement "
        "in Dual-output Second Language Speech Recognition"
    )
    framework: str = "L2-DO-ENTANGLE"
    affiliations: tuple[str, ...] = ("Hanyang University",)

    # Architecture (§2)
    encoder: str = "Conformer"
    auxiliary: str = "CTC on surface targets"
    do_params_m: float = 40.0
    so_conformer_params_m: float = 32.0

    # Loss weights (§2.2 Eq. 2)
    alpha_ctc: float = 0.2
    beta_surf: float = 0.5
    gamma_mean: float = 0.3

    # Table 2 — Conformer SO vs DO CER (%)
    ko_so_surface_cer: float = 11.14
    ko_so_meaning_cer: float = 1.60
    ko_do_surface_cer: float = 11.34
    ko_do_meaning_cer: float = 0.77
    en_so_surface_cer: float = 13.78
    en_so_meaning_cer: float = 3.87
    en_do_surface_cer: float = 15.08
    en_do_meaning_cer: float = 3.19

    # Figure 2 — stratified CER gap Δ = DO − SO at extremes
    en_surface_gap_ed0: float = 0.28
    en_surface_gap_ed_gt10: float = 6.72
    en_meaning_gap_ed0: float = -0.20
    en_meaning_gap_ed_gt10: float = -3.51

    # Table 3 — encoder SSO↔MSO CKA at final layer (layer 11)
    ko_encoder_sso_mso_layer11: float = 0.56
    en_encoder_sso_mso_layer11: float = 0.40

    # Table 4 — English decoder layer 7 (cross-task inversion)
    en_decoder_mso_mdo_layer7: float = 0.24
    en_decoder_mso_sdo_cross_layer7: float = 0.44
    en_decoder_sso_mdo_cross_layer7: float = 0.17
