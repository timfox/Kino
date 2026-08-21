"""Paper table anchors (Li et al., arXiv:2606.12657)."""

from __future__ import annotations

PAPER_ANCHORS = {
    "numosim_avg_visits": 7.2,
    "mobilitysyn_avg_visits": 8.7,
    "n_individuals": 1_200,
    "n_eval_trajectories": 34_000,
    "numosim_activity_types": 16,
    "mobilitysyn_activity_types": 6,
    "trajgenagent_gpu_hours": 1.67,
    "workflow_traj_success": 1.0,
    "freeform_traj_success": 0.098,
    "workflow_visit_success": 1.0,
    "freeform_visit_success": 0.593,
    "best_llm_temperature": 0.9,
    "best_verifier_failure_rate": 0.091,
}

TABLE_I_DATASETS = [
    {
        "dataset": "NumoSim",
        "total_daily_trajectories": 34_000,
        "avg_stay_points": 7.2,
        "n_individuals": 1_200,
        "n_activity_types": 16,
    },
    {
        "dataset": "MobilitySyn",
        "total_daily_trajectories": 34_000,
        "avg_stay_points": 8.7,
        "n_individuals": 1_200,
        "n_activity_types": 6,
    },
]

TABLE_II_NUMOSIM = [
    {"model": "GRU", "distance": 0.0111, "g_radius": 0.2557, "duration": 0.2145, "daily_loc": 0.1561, "i_rank": 0.0137, "g_rank": 0.0159, "transition": 0.0156},
    {"model": "Geo-Llama", "distance": 0.0075, "g_radius": 0.2361, "duration": 0.0028, "daily_loc": 0.0128, "i_rank": 0.0001, "g_rank": 0.0001, "transition": 0.0087},
    {"model": "TrajGenAgent", "distance": 0.0006, "g_radius": 0.0993, "duration": 0.0155, "daily_loc": 0.2117, "i_rank": 0.0002, "g_rank": 0.0002, "transition": 0.0075},
]

TABLE_II_MOBILITYSYN = [
    {"model": "Geo-Llama", "distance": 0.0268, "g_radius": 0.5528, "duration": 0.0241, "daily_loc": 0.1209, "i_rank": 0.0005, "g_rank": 0.0005, "transition": 0.0078},
    {"model": "Geo-CETRA", "distance": 0.0276, "g_radius": 0.5784, "duration": 0.0319, "daily_loc": 0.1573, "i_rank": 0.0006, "g_rank": 0.0006, "transition": 0.0083},
    {"model": "TrajGenAgent", "distance": 0.0000, "g_radius": 0.0051, "duration": 0.1308, "daily_loc": 0.0000, "i_rank": 0.0003, "g_rank": 0.0003, "transition": 0.0000},
]

TABLE_III_ANOMALY = [
    {
        "dataset": "NumoSim",
        "model": "TrajGenAgent",
        "bestad_auroc": 0.5008,
        "bestad_ap": 0.5255,
        "icad_visit_auroc": 0.5368,
        "icad_visit_ap": 0.5690,
        "icad_ind_auroc": 0.6398,
    },
    {
        "dataset": "MobilitySyn",
        "model": "TrajGenAgent",
        "bestad_auroc": 0.6817,
        "bestad_ap": 0.6318,
        "icad_visit_auroc": 0.6761,
        "icad_visit_ap": 0.6342,
        "icad_ind_auroc": 0.7194,
    },
]

TABLE_IV_GPU_HOURS = [
    {"model": "GRU", "gpu_hours": 1.25},
    {"model": "TrajGenAgent", "gpu_hours": 1.67},
    {"model": "LSTM", "gpu_hours": 1.83},
    {"model": "Geo-CETRA", "gpu_hours": 3.38},
    {"model": "SeqGAN", "gpu_hours": 20.62},
    {"model": "Geo-Llama", "gpu_hours": 24.77},
]

TABLE_V_TOOL_STABILITY = [
    {"strategy": "Free-form tool calling", "trajectory_success": 0.098, "visit_success": 0.593},
    {"strategy": "Deterministic workflow", "trajectory_success": 1.0, "visit_success": 1.0},
]

TABLE_VIII_TEMPERATURE = [
    {"temperature": 0.5, "total_fr": 0.143, "schema_fr": 0.018, "constraint_fr": 0.125},
    {"temperature": 0.9, "total_fr": 0.091, "schema_fr": 0.036, "constraint_fr": 0.055},
    {"temperature": 1.5, "total_fr": 0.283, "schema_fr": 0.152, "constraint_fr": 0.131},
]

TABLE_VI_KINEMATICS = [
    {"dataset": "NumoSim", "variant": "with_kinematics", "distance": 0.0006, "g_radius": 0.0993, "duration": 0.0155, "daily_loc": 0.2117, "i_rank": 0.0002, "g_rank": 0.0002, "transition": 0.0075},
    {"dataset": "NumoSim", "variant": "without_kinematics", "distance": 0.0028, "g_radius": 0.1508, "duration": 0.0198, "daily_loc": 0.2476, "i_rank": 0.0006, "g_rank": 0.0006, "transition": 0.0077},
    {"dataset": "MobilitySyn", "variant": "with_kinematics", "distance": 0.0000, "g_radius": 0.0051, "duration": 0.1308, "daily_loc": 0.0000, "i_rank": 0.0003, "g_rank": 0.0003, "transition": 0.0000},
    {"dataset": "MobilitySyn", "variant": "without_kinematics", "distance": 0.0000, "g_radius": 0.0004, "duration": 0.3732, "daily_loc": 0.0000, "i_rank": 0.0001, "g_rank": 0.0001, "transition": 0.0000},
]

TABLE_VII_KINEMATICS_ANOMALY = [
    {"dataset": "NumoSim", "variant": "with_kinematics", "bestad_auroc": 0.5008, "icad_visit_ap": 0.5255, "icad_visit_auroc": 0.5368, "icad_ind_ap": 0.5690, "icad_ind_auroc": 0.6398},
    {"dataset": "NumoSim", "variant": "without_kinematics", "bestad_auroc": 0.5104, "icad_visit_ap": 0.6692, "icad_visit_auroc": 0.6483, "icad_ind_ap": 0.9034, "icad_ind_auroc": 0.9143},
    {"dataset": "MobilitySyn", "variant": "with_kinematics", "bestad_auroc": 0.6817, "icad_visit_ap": 0.6318, "icad_visit_auroc": 0.6761, "icad_ind_ap": 0.6342, "icad_ind_auroc": 0.7194},
    {"dataset": "MobilitySyn", "variant": "without_kinematics", "bestad_auroc": 0.7400, "icad_visit_ap": 0.7143, "icad_visit_auroc": 0.6827, "icad_ind_ap": 0.9922, "icad_ind_auroc": 0.9925},
]


def benchmarks_bundle() -> dict[str, object]:
    return {
        "anchors": PAPER_ANCHORS,
        "table_i_datasets": TABLE_I_DATASETS,
        "table_ii_numosim": TABLE_II_NUMOSIM,
        "table_ii_mobilitysyn": TABLE_II_MOBILITYSYN,
        "table_iii_anomaly": TABLE_III_ANOMALY,
        "table_iv_gpu_hours": TABLE_IV_GPU_HOURS,
        "table_v_tool_stability": TABLE_V_TOOL_STABILITY,
        "table_vi_kinematics": TABLE_VI_KINEMATICS,
        "table_vii_kinematics_anomaly": TABLE_VII_KINEMATICS_ANOMALY,
        "table_viii_temperature": TABLE_VIII_TEMPERATURE,
    }
