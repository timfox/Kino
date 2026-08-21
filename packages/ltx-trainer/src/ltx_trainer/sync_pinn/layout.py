"""Pipeline stages and stub limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "Joint NN parameterization Nx(t), Nu(t) with shaping h(t) enforcing ICs (Eq. 2)",
    "Physics-informed residual r(t) = dx/dt − f(p,x,u) at collocation times (Eq. 3)",
    "Composite loss L = Ldyn + Lic + Lcontrol + Lreg (Eq. 4)",
    "Kuramoto instantiation: order parameter R(t), persistence R(t)≥R* for t≥t* (Eqs. 6–10)",
    "Report instantaneous cost P(t) and integrated E post hoc (Eqs. 13–14)",
    "Benchmark vs phase feedback and frequency compensation; extend to Sakaguchi (Eq. 18)",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — no full PINN training loop, autograd graph, or Chroma data.",
    "Toy Euler rollouts and analytic loss terms only; not a reproduction of Fig. 2 trajectories.",
    "Offline trajectory design — no online adaptive feedback or model mismatch guarantees.",
)
