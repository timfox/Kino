"""High-level failure taxonomy: acting, atmosphere, cinematography (Table 2)."""

from __future__ import annotations

from typing import Any

# Major categories → sub-dimensions (Table 2).
CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("acting", "EP", "Emotional Performance"),
    ("acting", "MP", "Motion Performance"),
    ("acting", "DP", "Dialogue Performance"),
    ("acting", "IP", "Interaction Performance"),
    ("atmosphere", "MC", "Mood Construction"),
    ("atmosphere", "EC", "Environmental Coherence"),
    ("atmosphere", "SD", "Soundscape Design"),
    ("cinematography", "IC", "Intra-shot Camera"),
    ("cinematography", "IG", "Inter-shot Grammar"),
    ("cinematography", "CT", "Continuity"),
)

SUB_DIMENSION_BY_CODE: dict[str, dict[str, str]] = {
    code: {"category": cat, "name": name, "code": code}
    for cat, code, name in CATEGORIES
}

# Fine-grained failure modes (45 total; paper Table 2 / Fig. 3 / Appendix G).
FAILURE_MODES: tuple[dict[str, str], ...] = (
    # Acting — EP
    {"id": "facial_expression_inconsistency", "sub_dim": "EP", "label": "Facial Expression Inconsistency"},
    {"id": "expression_exaggeration", "sub_dim": "EP", "label": "Expression Exaggeration"},
    {"id": "multimodal_expressive_inconsistency", "sub_dim": "EP", "label": "Multimodal Expressive Inconsistency"},
    {"id": "vocal_emotional_delivery_failure", "sub_dim": "EP", "label": "Vocal Emotional Delivery Failure"},
    {"id": "co_speech_gesture_inconsistency", "sub_dim": "EP", "label": "Co-Speech Gesture Inconsistency"},
    {"id": "emotion_driven_motivation_shift", "sub_dim": "EP", "label": "Emotion-Driven Motivation Shift"},
    # Acting — MP
    {"id": "robotic_movement", "sub_dim": "MP", "label": "Robotic Movement"},
    {"id": "over_smoothed_movement", "sub_dim": "MP", "label": "Over-smoothed Movement"},
    {"id": "action_exaggeration", "sub_dim": "MP", "label": "Action Exaggeration"},
    {"id": "action_missing", "sub_dim": "MP", "label": "Action Missing"},
    {"id": "action_not_matching_scripts", "sub_dim": "MP", "label": "Action Not Matching Scripts"},
    {"id": "plot_driven_intention_failure", "sub_dim": "MP", "label": "Plot-Driven Intention Failure"},
    # Acting — DP
    {"id": "speech_mode_confusion", "sub_dim": "DP", "label": "Speech Mode Confusion"},
    {"id": "dialogue_content_deviation", "sub_dim": "DP", "label": "Dialogue Content Deviation"},
    {"id": "speaker_identity_error", "sub_dim": "DP", "label": "Speaker Identity Error"},
    # Acting — IP
    {"id": "interaction_eyeline_failure", "sub_dim": "IP", "label": "Interaction Eyeline Failure"},
    {"id": "reaction_timing_failure", "sub_dim": "IP", "label": "Reaction Timing Failure"},
    {"id": "dialogue_turn_taking_failure", "sub_dim": "IP", "label": "Dialogue Turn-Taking Failure"},
    {"id": "proxemics_relationship_failure", "sub_dim": "IP", "label": "Proxemics Relationship Failure"},
    {"id": "character_targeting_error", "sub_dim": "IP", "label": "Character Targeting Error"},
    {"id": "environmental_awareness_missing", "sub_dim": "IP", "label": "Environmental Awareness Missing"},
    # Atmosphere — MC
    {"id": "lighting_emotion_conflict", "sub_dim": "MC", "label": "Lighting Emotion Conflict"},
    {"id": "color_emotion_conflict", "sub_dim": "MC", "label": "Color Emotion Conflict"},
    {"id": "scene_music_emotion_conflict", "sub_dim": "MC", "label": "Scene–Music Emotion Conflict"},
    {"id": "visual_mood_drift", "sub_dim": "MC", "label": "Visual Mood Drift"},
    {"id": "music_tone_drift", "sub_dim": "MC", "label": "Music Tone Drift"},
    # Atmosphere — EC
    {"id": "artificial_environment", "sub_dim": "EC", "label": "Artificial Environment"},
    {"id": "lack_of_environmental_depth", "sub_dim": "EC", "label": "Lack of Environmental Depth"},
    {"id": "scene_consistency_drift", "sub_dim": "EC", "label": "Scene Consistency Drift"},
    {"id": "missing_scene_transition", "sub_dim": "EC", "label": "Missing Scene Transition"},
    {"id": "artificial_environment_transition", "sub_dim": "EC", "label": "Artificial Environment Transition"},
    # Atmosphere — SD
    {"id": "missing_ambient_sound", "sub_dim": "SD", "label": "Missing Ambient Sound"},
    {"id": "overpowering_background_audio", "sub_dim": "SD", "label": "Overpowering Background Audio"},
    {"id": "missing_scripted_sound", "sub_dim": "SD", "label": "Missing Scripted Sound"},
    # Cinematography — IC
    {"id": "camera_action_misalignment", "sub_dim": "IC", "label": "Camera-Action Misalignment"},
    {"id": "attention_guidance_failure", "sub_dim": "IC", "label": "Attention Guidance Failure"},
    {"id": "focus_control_failure", "sub_dim": "IC", "label": "Focus Control Failure"},
    {"id": "camera_motivation_issues", "sub_dim": "IC", "label": "Camera Motivation Issues"},
    {"id": "blocking_issues", "sub_dim": "IC", "label": "Blocking Issues"},
    # Cinematography — IG
    {"id": "shot_progression_issues", "sub_dim": "IG", "label": "Shot Progression Issues"},
    {"id": "thirty_degree_rule_violation", "sub_dim": "IG", "label": "30-Degree Rule Violation"},
    {"id": "one_eighty_rule_violation", "sub_dim": "IG", "label": "180-Degree Rule Violation"},
    # Cinematography — CT
    {"id": "action_continuity_break", "sub_dim": "CT", "label": "Action Continuity Break"},
    {"id": "spatial_continuity_break", "sub_dim": "CT", "label": "Spatial Continuity Break"},
    {"id": "character_continuity_break", "sub_dim": "CT", "label": "Character Continuity Break"},
)

assert len(FAILURE_MODES) == 45

FAILURE_MODE_BY_ID: dict[str, dict[str, str]] = {m["id"]: m for m in FAILURE_MODES}


def failure_modes_for_sub_dim(code: str) -> list[dict[str, str]]:
    return [m for m in FAILURE_MODES if m["sub_dim"] == code.upper()]


def taxonomy_card() -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, str]]] = {"acting": [], "atmosphere": [], "cinematography": []}
    for cat, code, name in CATEGORIES:
        by_cat[cat].append(
            {
                "code": code,
                "name": name,
                "failure_mode_count": len(failure_modes_for_sub_dim(code)),
            }
        )
    return {
        "categories": list(by_cat.keys()),
        "sub_dimensions": by_cat,
        "n_failure_modes": len(FAILURE_MODES),
    }
