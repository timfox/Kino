"""Unified global + event-level annotation schema (Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventAnnotation:
    event_id: int
    time_range: str
    action: str
    completion_flag: str
    visual_description: str
    audio_expectation: str
    qa_questions: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BenchmarkCase:
    case_id: str
    task: str
    scenario: str
    complexity: str
    language: str
    global_description: str
    events: list[EventAnnotation]
    reference_image: str | None = None
    reference_video: str | None = None
    identity_tracking: list[str] = field(default_factory=list)
    physical_constraints: list[str] = field(default_factory=list)


def case_to_dict(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "task": case.task,
        "scenario": case.scenario,
        "complexity": case.complexity,
        "language": case.language,
        "global_description": case.global_description,
        "reference_image": case.reference_image,
        "reference_video": case.reference_video,
        "identity_tracking": case.identity_tracking,
        "physical_constraints": case.physical_constraints,
        "events": [
            {
                "event_id": e.event_id,
                "time_range": e.time_range,
                "action": e.action,
                "completion_flag": e.completion_flag,
                "visual_description": e.visual_description,
                "audio_expectation": e.audio_expectation,
                "qa_questions": e.qa_questions,
            }
            for e in case.events
        ],
    }


def example_t2av_performance_ads_l4() -> BenchmarkCase:
    """Appendix C.1 — skincare two-actor product demo (abbreviated)."""
    return BenchmarkCase(
        case_id="t2av_perf_ads_l4_demo",
        task="T2AV",
        scenario="Performance Ads",
        complexity="L4",
        language="en",
        global_description=(
            "A 60-second skincare advertisement: Xiao-Lin demonstrates Water Glow Serum "
            "to troubled Xiao-Ya, culminating in a product hero shot."
        ),
        events=[
            EventAnnotation(
                1,
                "0-18s",
                "Recommend product",
                "Xiao-Ya holds the bottle",
                "Two women at bright window; product handoff",
                "Ambient room tone; soft dialogue",
                qa_questions=[
                    {"id": "subject", "prompt": "Are two women and the skincare product visible?"},
                    {"id": "action", "prompt": "Does one woman hand the product to the other?"},
                    {"id": "scene", "prompt": "Is the setting a bright indoor window area?"},
                ],
            ),
            EventAnnotation(
                2,
                "18-35s",
                "Demonstrate texture",
                "Serum spreads on palm with water-burst effect",
                "Close-up pump and spread on palm",
                "Subtle squeeze SFX",
            ),
            EventAnnotation(
                3,
                "35-50s",
                "Recipient tries product",
                "Xiao-Ya reacts with surprise smile",
                "Essence on Xiao-Ya's palm",
                "Light sparkle SFX",
            ),
            EventAnnotation(
                4,
                "50-60s",
                "Product hero",
                "Bottle centered with soft glow",
                "Packaging close-up",
                "Music swell",
            ),
        ],
        identity_tracking=[
            "subject 1: Xiao-Lin, confident skincare expert",
            "subject 2: Xiao-Ya, gentle personality",
        ],
        physical_constraints=["Product label must remain consistent", "Window lighting stable"],
    )


def example_i2av_performance_ads_l4() -> BenchmarkCase:
    """Appendix C.3 — Apple Watch product flat-lay to lifestyle video."""
    return BenchmarkCase(
        case_id="i2av_perf_ads_l4_watch",
        task="I2AV",
        scenario="Performance Ads",
        complexity="L4",
        language="en",
        global_description=(
            "A 60-second performance ad starting from a desk flat-lay of an Apple Watch; "
            "hand interaction, fitness rings, face montage, return to hero product shot."
        ),
        reference_image="apple_watch_desk_flatlay.jpg",
        events=[
            EventAnnotation(
                1,
                "0-9s",
                "Establish desk scene",
                "Notification animates on watch face",
                "Overhead desk: watch, notebook, pen, iPod",
                "Soft ambient desk tone",
            ),
            EventAnnotation(
                2,
                "9-20s",
                "Hand picks up watch",
                "Watch on wrist",
                "Hand enters frame, puts watch on",
                "Fabric rustle",
            ),
            EventAnnotation(
                3,
                "20-35s",
                "Fitness tracking demo",
                "Activity rings fill on watch face",
                "Close-up watch UI animation",
                "UI tick SFX",
            ),
            EventAnnotation(
                4,
                "35-48s",
                "Face montage",
                "Multiple complications shown",
                "Quick-cut watch faces",
                "Upbeat music",
            ),
            EventAnnotation(
                5,
                "48-60s",
                "Brand CTA",
                "Logo and tagline on desk composition",
                "Return to original flat-lay angle",
                "Music resolve",
            ),
        ],
        identity_tracking=["Apple Watch silver aluminum case", "Gray-white woven band"],
        physical_constraints=[
            "Warm directional lighting from left",
            "Desk objects must persist",
            "Watch band color stable",
        ],
    )


def example_v2av_content_creator_l4() -> BenchmarkCase:
    """Appendix C.2 — train-station short-film continuation."""
    return BenchmarkCase(
        case_id="v2av_content_creator_l4_missed_connection",
        task="V2AV",
        scenario="Content-Creator",
        complexity="L4",
        language="en",
        global_description=(
            "After the reference clip of a man running through a train station, continue with "
            "collision, eye contact, romantic montage, departure, and title card THE MISSED CONNECTION."
        ),
        reference_video="train_station_run_12s.mp4",
        events=[
            EventAnnotation(
                2,
                "8-13s",
                "Collision",
                "Papers scatter on platform",
                "Briefcase opens; papers fly",
                "Impact and paper flutter SFX",
            ),
            EventAnnotation(
                3,
                "13-17s",
                "Eye contact",
                "Man and woman meet eyes",
                "Kneeling man looks up at woman in leather jacket",
                "Platform ambience",
            ),
            EventAnnotation(
                4,
                "17-32s",
                "Romantic montage",
                "Fantasy sequence completes",
                "Hug, proposal, wedding, house keys, pregnancy",
                "Dreamy score",
            ),
            EventAnnotation(
                5,
                "32-40s",
                "Woman boards train",
                "Train doors close",
                "Woman enters train; man stays",
                "Train departure",
            ),
            EventAnnotation(
                6,
                "40-52s",
                "Alone on platform",
                "Train leaves frame",
                "Man stands with scattered papers",
                "Distant train fade",
            ),
            EventAnnotation(
                7,
                "52-65s",
                "Title card",
                "THE MISSED CONNECTION visible",
                "Static title typography",
                "Music sting",
            ),
        ],
        identity_tracking=[
            "subject 1: reddish-blond man, dark suit, brown briefcase",
            "subject 2: dark hair, black leather jacket, red lipstick",
        ],
        physical_constraints=[
            "Man does not board train",
            "Papers remain on platform at end",
            "Outdoor elevated station architecture",
        ],
    )


def build_generation_prompt(case: BenchmarkCase, *, include_audio: bool = True) -> dict[str, str]:
    """Appendix D — task-specific generation inputs from structured annotation."""
    audio_parts = "; ".join(e.audio_expectation for e in case.events if e.audio_expectation)
    out: dict[str, str] = {"video_prompt": case.global_description}
    if include_audio and audio_parts:
        out["audio_prompt"] = audio_parts
    if case.task == "I2AV" and case.reference_image:
        out["reference_image"] = case.reference_image
    if case.task == "V2AV" and case.reference_video:
        out["reference_video"] = case.reference_video
        cont_visual = "; ".join(e.visual_description for e in case.events)
        cont_audio = "; ".join(e.audio_expectation for e in case.events)
        out["video_prompt"] = f"After the reference video, the scene continues with: {cont_visual}"
        if include_audio:
            out["audio_prompt"] = f"After the reference video, the audio continues with: {cont_audio}"
    return out


def challenging_case_summaries() -> list[dict[str, str]]:
    """Appendix C.4 — representative hard cases (metadata only)."""
    return [
        {
            "case_id": "t2av_real_perf_18ev_nail_art",
            "task": "T2AV",
            "scenario": "Performance Ads",
            "complexity": "L3",
            "challenge": "18 events; product colors and hand identity across rapid transitions",
        },
        {
            "case_id": "t2av_real_creator_13ev_drama",
            "task": "T2AV",
            "scenario": "Content-Creator",
            "complexity": "L4",
            "challenge": "Multi-actor emotional arcs; identity drift after event 6–7",
        },
        {
            "case_id": "t2av_real_brand_15ev_aerial",
            "task": "T2AV",
            "scenario": "Brand Ads",
            "complexity": "L4",
            "challenge": "Continuous drone motion, landscape geometry, text overlays",
        },
    ]
