"""Hard-negative and evaluation-dimension taxonomies (Sec. 3.2, 3.4, Fig. 4, Tables 3–5)."""

from __future__ import annotations

# Audio–video negative construction codes (Table 3).
AV_NEGATIVE_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("Basic Semantic Negative", "random_mismatch"),
    ("High-level Semantic Negative", "semantic_mismatch"),
    ("High-precision Temporal Negative", "time_shift_micro"),
    ("Temporal Negative", "time_shift_medium"),
    ("Temporal-Physical Negative", "speed_change"),
    ("Speaker/Physical Negative", "pitch_shift"),
    ("Acoustic Scene Negative", "noise_addition"),
    ("Acoustic Structural Negative", "filter"),
)

# Video–text dimensions (Table 4 excerpt — primary dimensions).
VT_DIMENSIONS: tuple[str, ...] = (
    "Appearance",
    "Emotion & Expression",
    "Age & Gender",
    "Social Relation",
    "Counting",
    "Motion",
    "Interaction",
    "State",
    "Spatial",
    "Saliency",
    "Temporal",
    "Camera",
    "Logical",
    "World Knowledge",
)

# Audio–text primary categories (Table 5 excerpt).
AT_PRIMARY_CATEGORIES: tuple[str, ...] = (
    "Speech Attributes",
    "Speech Content",
    "Emotion & Pragmatics",
    "Counting & Degree",
    "Sound Effects",
    "Action–Sound Align.",
    "Acoustic Environ.",
    "Music Attributes",
    "Audio Saliency",
    "Temporal Structure",
    "Logical (Audio)",
    "World Knowledge",
    "Language & Annot.",
)

# Ten suite dimensions (Sec. 3.4).
SUITE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("Cross-modal", "AT consistency (SFT Qwen2-Audio)"),
    ("Cross-modal", "VT consistency (SFT Qwen2.5-Omni)"),
    ("Cross-modal", "AV consistency (SFT Qwen2.5-Omni)"),
    ("Cross-modal", "Lip sync (SyncNet / LatentSync)"),
    ("Unimodal", "Speech content (Whisper-large-v3: S_comp, S_acc, S_hall)"),
    ("Unimodal", "Speech realism (DF_Arena)"),
    ("Unimodal", "Audio quality (NISQAv2 MOS)"),
    ("Unimodal", "Audio aesthetics (Audiobox-Aesthetics)"),
    ("Unimodal", "Video quality (DOVER++)"),
    ("Unimodal", "Video aesthetics (LAION-Aesthetics)"),
)
