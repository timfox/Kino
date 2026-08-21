"""Reward-model instruction templates (Appendix A.5, abbreviated)."""

from __future__ import annotations

IMAGE_REWARD_WITH_REF = """You are an expert in image captioning. Compare the Generated Caption to the Reference Caption using the Input Image as ground truth.
Score Correctness, Completeness, and Text Quality each 0-10.
Return JSON: {"Analysis": "...", "Correctness": n, "Completeness": n, "Text Quality": n}
"""

VIDEO_GLOBAL_REWARD = """You are an expert video-description quality evaluator. Rate Reasonability, Correctness, and Completeness 0-10 using frames as ground truth and Reference Description as auxiliary context only.
Return JSON with Analysis and the three scores.
"""

DENSE_CAPTION_INSTRUCTION = """Describe this visual content with dense, fact-level detail suitable for witness-adjudicator training.
Cover main subjects, attributes, colors, spatial relations, actions, and scene context.
Avoid meta-commentary (e.g. "all elements described"). Be factual and specific; do not invent objects not visible.
"""
