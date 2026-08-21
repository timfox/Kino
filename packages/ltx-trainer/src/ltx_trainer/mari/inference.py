"""End-to-end MARI inference on hidden states (numpy reference)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.mari.adapters import MultiAdapterBank
from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.energy import EnergyGate, calibrate_energy_threshold, propagation_energy, probe_delta, simulate_post_injection_layers
from ltx_trainer.mari.low_rank import LowRankAdapter, init_adapter
from ltx_trainer.mari.pca import PCASubspace, fit_pca


@dataclass
class MARIInferenceResult:
    prediction: int
    adapter_index: int
    energy: float
    applicable: bool
    alpha: float
    option_logits: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": int(self.prediction),
            "adapter_index": int(self.adapter_index),
            "energy": round(self.energy, 6),
            "applicable": bool(self.applicable),
            "alpha": float(self.alpha),
            "option_logits": self.option_logits.tolist(),
        }


@dataclass
class MARIPipeline:
    cfg: MARIConfig
    bank: MultiAdapterBank
    probe: LowRankAdapter
    pca: PCASubspace
    gate: EnergyGate
    option_embeddings: np.ndarray = field(repr=False)

    @classmethod
    def from_config(cls, cfg: MARIConfig | None = None, seed: int = 0) -> MARIPipeline:
        cfg = cfg or MARIConfig()
        rng = np.random.default_rng(seed)
        bank = MultiAdapterBank.create(cfg, seed=seed)
        probe = init_adapter(cfg.hidden_dim, cfg.probe_rank, rng)
        # Synthetic calibration hiddens for PCA
        cal_h = rng.standard_normal((32, cfg.hidden_dim)) * 0.1
        pca = fit_pca(cal_h, cfg.pca_rank)
        gate = EnergyGate(
            threshold=0.0,
            alpha_probe=cfg.probe_alpha,
            alpha_full=cfg.alpha_full,
            alpha_safe=cfg.alpha_safe,
        )
        opts = rng.standard_normal((4, cfg.hidden_dim))
        opts /= np.linalg.norm(opts, axis=1, keepdims=True) + 1e-9
        return cls(cfg=cfg, bank=bank, probe=probe, pca=pca, gate=gate, option_embeddings=opts)

    def calibrate_gate(
        self,
        hiddens: np.ndarray,
        applicable: np.ndarray,
        *,
        rho: float | None = None,
    ) -> float:
        rho = self.cfg.target_rejection_rate if rho is None else rho
        energies = []
        for h in hiddens:
            d = probe_delta(self.probe, h, alpha_probe=self.cfg.probe_alpha)
            base, probe = simulate_post_injection_layers(
                h, d, num_layers=self.cfg.num_post_layers
            )
            energies.append(propagation_energy(base, probe))
        tau = calibrate_energy_threshold(np.array(energies), applicable, rho=rho)
        self.gate = EnergyGate(
            threshold=tau,
            alpha_probe=self.cfg.probe_alpha,
            alpha_full=self.cfg.alpha_full,
            alpha_safe=self.cfg.alpha_safe,
        )
        return tau

    def forward_mc(
        self,
        h: np.ndarray,
        *,
        option_embeddings: np.ndarray | None = None,
    ) -> MARIInferenceResult:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        opts = option_embeddings if option_embeddings is not None else self.option_embeddings

        d_probe = probe_delta(self.probe, h, alpha_probe=self.cfg.probe_alpha)
        base_layers, probe_layers = simulate_post_injection_layers(
            h, d_probe, num_layers=self.cfg.num_post_layers
        )
        energy = propagation_energy(base_layers, probe_layers)
        alpha = self.gate.actuation_alpha(energy)
        applicable = self.gate.is_applicable(energy)

        if alpha <= 1e-9:
            logits = opts @ h
            pred = int(np.argmax(logits))
            return MARIInferenceResult(
                prediction=pred,
                adapter_index=-1,
                energy=energy,
                applicable=applicable,
                alpha=alpha,
                option_logits=logits,
            )

        k = self.bank.infer_route(h, opts, alpha=alpha)
        h_edit = self.bank.adapters[k].intervene(h, gamma=self.bank.gamma, alpha=alpha)
        logits = opts @ h_edit
        pred = int(np.argmax(logits))
        return MARIInferenceResult(
            prediction=pred,
            adapter_index=k,
            energy=energy,
            applicable=applicable,
            alpha=alpha,
            option_logits=logits,
        )
