"""MM '25 reproducibility companion artifacts (arXiv:2605.26313, DOI 10.1145/3746027.3759199)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.swarical.aruco_pose import aruco_reproduction_guide
from ltx_trainer.swarical.config import SwaricalConfig
from ltx_trainer.swarical.datasets import list_mesh_datasets
from ltx_trainer.swarical.online_sim import compare_localization_modes, run_small_scale_experiment


def companion_paper_metadata() -> dict[str, Any]:
    return {
        "title": (
            "Reproducibility Companion Paper: Swarical: An Integrated Hierarchical Approach "
            "to Localizing Flying Light Specks"
        ),
        "venue": "ACM MM '25, Dublin",
        "doi": "10.1145/3746027.3759199",
        "arxiv": "arXiv:2605.26313",
        "original_paper_doi": "10.1145/3664647.3681080",
        "license": "CC BY 4.0",
        "authors": [
            "Hamed Alimohammadzadeh",
            "Shahram Ghandeharizadeh",
            "Federico Cunico",
            "Joshua Springer",
        ],
    }


def artifact_manifest() -> dict[str, Any]:
    """Repositories, Docker workflow, and cloud paths from companion Sec. 1–2."""
    cfg = SwaricalConfig()
    return {
        "companion": companion_paper_metadata(),
        "repositories": [
            {
                "name": "Swarical",
                "url": cfg.github,
                "role": "Offline planner + online ISR/HC/RSF + reproduction.ipynb",
            },
            {
                "name": "aruco-pose-estimation",
                "url": "https://github.com/flyinglightspeck/aruco-pose-estimation",
                "role": "Raspberry Pi Camera Module 3 + ArUco relative pose (Sec. 3)",
            },
        ],
        "docker_small_scale": {
            "steps": [
                "git clone https://github.com/flyinglightspeck/Swarical.git",
                "cd Swarical && bash init.sh",
                "Open Jupyter URL on http://127.0.0.1:8888/...",
                "Run reproduction.ipynb (planner → dead reckoning → 16-FLS localization)",
            ],
            "notebook": "reproduction.ipynb",
            "jupyter_port": 8888,
            "runtime_note": "16 FLS processes; default ~30–60 s; Hausdorff < 1 cm in few seconds",
        },
        "cloud_large_scale": {
            "providers": ["CloudBank", "Amazon AWS via CloudLab"],
            "readmes": [
                "https://github.com/flyinglightspeck/Swarical/blob/main/CloudLab_README.md",
                "https://github.com/flyinglightspeck/Swarical/blob/main/AWS_README.md",
            ],
            "requirement": "Cluster cores ≥ F (one FLS process per core)",
            "skateboard_fls": cfg.skateboard_fls_count,
            "protocol": "UDP broadcast; set PLATFORM=cloudlab in constants.py on CloudLab",
        },
        "plot_artifacts": {
            "swaricalplots_nb": "Mathematica notebook for Figs. 10–12, 13–15",
            "cameraplots_nb": "Mathematica notebook for camera Figs. 6–9 (aruco repo)",
            "thousand_core_logs": "Pre-downloaded in Docker image for Step 21 animations",
        },
        "datasets": list_mesh_datasets(),
        "aruco": aruco_reproduction_guide(),
    }


def reproduction_walkthrough() -> dict[str, Any]:
    """Ordered steps matching companion Sec. 2.1 (small scale) + Sec. 2.2 (online)."""
    small = run_small_scale_experiment()
    modes = compare_localization_modes()
    return {
        "manifest": artifact_manifest(),
        "small_scale_demo": small,
        "mode_comparison_4x4": {
            "ranking": modes["ranking_by_final_hd"],
            "isr_hd_below_1cm_step": modes["runs"]["ISR"]["hd_below_1cm_at_step"],
        },
        "checklist": [
            "Clone Swarical; build Docker via init.sh",
            "Step through reproduction.ipynb — seven meshes + swarm trees (<2 s)",
            "Run 16-FLS decentralized localization; copy log for animation cell",
            "Optional: thousand-core Skateboard logs (ISR/HC/RSF) bundled in image",
            "Clone aruco-pose-estimation on Raspberry Pi 5 for camera Sec. 3",
        ],
    }


def evaluation_smoke() -> dict[str, Any]:
    """Extended smoke for validate_paper_stubs / run_paper_stub_smoke."""
    small = run_small_scale_experiment(max_iterations=20)
    return {
        "companion_doi": companion_paper_metadata()["doi"],
        "n_fls_small_scale": small["n_fls"],
        "final_hausdorff_cm": small["final_hausdorff_cm"],
        "hd_below_1cm": small["hd_below_1cm_at_step"] is not None,
    }
