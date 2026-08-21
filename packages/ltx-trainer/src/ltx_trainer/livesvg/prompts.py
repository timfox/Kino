"""I2V and Gemini filtering prompt templates (Sec. 3.3, suppl. F)."""

from __future__ import annotations


def i2v_motion_prompt(motion: str) -> str:
    """Flat 2D vector-style animation prompt for target video generation."""
    return (
        f"Animate this image as flat 2D vector-style animation of {motion}. "
        "Style: pure 2D motion graphics, solid block colors, crisp edges, no gradients, "
        "no shading, and zero 3D depth. Motion: restrict all movement strictly to the 2D "
        "image plane using simple articulated or puppet-style motion. Consistency: preserve "
        "the original geometric shapes and exact colors; maintain the object's current facing "
        "direction; do not add new details or remove existing details. Environment: solid white "
        "background with no new scene content."
    )


def gemini_stage1_rubric() -> dict[str, str]:
    """First-stage viability scorer criteria (suppl. Gemini template)."""
    return {
        "no_3d_rotation": "No 3D rotation revealing hidden geometry; 2D morphing allowed.",
        "no_new_elements": "No semantic elements beyond the first frame.",
        "flat_colors": "Constant flat fills; no lighting or color grading.",
        "physical_continuity": "No sudden appearance/disappearance except ordinary occlusion.",
        "overall": "Viable fixed-topology, fixed-color SVG tracking target.",
    }


def gemini_stage2_instruction() -> str:
    return (
        "Rank candidate videos by suitability as ground-truth targets for downstream SVG optimization. "
        "Favor least color drift, cleanest 2D motion, fewest invented elements, and most stable structure. "
        "A slightly less dynamic but clean video beats a dramatic artifact-heavy clip."
    )


def method_capability_table() -> dict[str, dict[str, str]]:
    """Table 1 — practical capability comparison (paper)."""
    return {
        "Vector Prism": {
            "unconstrained_motion": "partial",
            "skeleton_free": "yes",
            "open_domain": "yes",
            "complex_scenes": "no",
        },
        "LiveSketch": {
            "unconstrained_motion": "yes",
            "skeleton_free": "yes",
            "open_domain": "yes",
            "complex_scenes": "no",
        },
        "AniClipart": {
            "unconstrained_motion": "no",
            "skeleton_free": "no",
            "open_domain": "no",
            "complex_scenes": "no",
        },
        "FlexiClip": {
            "unconstrained_motion": "no",
            "skeleton_free": "no",
            "open_domain": "no",
            "complex_scenes": "no",
        },
        "LINR-Bridge": {
            "unconstrained_motion": "no",
            "skeleton_free": "yes",
            "open_domain": "no",
            "complex_scenes": "no",
        },
        "LiveSVG": {
            "unconstrained_motion": "yes",
            "skeleton_free": "yes",
            "open_domain": "yes",
            "complex_scenes": "yes",
        },
    }
