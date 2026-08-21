"""Spatial QA templates (Appendix E)."""

from __future__ import annotations

import math

from ltx_trainer.spacedg.schema import AnswerFormat, QAPair, QuestionType, SpatialTaskGroup
from ltx_trainer.spacedg.scene import ObjectInstance, SceneAnnotation


def format_camera_translation_mcq() -> QAPair:
    return QAPair(
        question=(
            "What is the primary camera motion direction from view A to view B in view A's coordinate?"
        ),
        answer="C",
        question_type=QuestionType.CAMERA_TRANSLATION,
        task_group=SpatialTaskGroup.CAMERA_CENTRIC,
        answer_format=AnswerFormat.MCQ,
        options=["left-backward", "left-forward", "right-forward", "right-backward"],
        num_views=2,
    )


def format_object_counting(label: str, count: int) -> QAPair:
    return QAPair(
        question=f"Count the number of {label} objects in the scene.",
        answer=str(count),
        question_type=QuestionType.OBJECT_COUNTING,
        task_group=SpatialTaskGroup.OBJECT_CENTRIC,
        answer_format=AnswerFormat.NUMERIC,
        num_views=1,
    )


def format_inter_object_distance(obj_a: ObjectInstance, obj_b: ObjectInstance, dist_m: float) -> QAPair:
    return QAPair(
        question=(
            f"Estimate the 3D metric distance (in meters) between the centers of "
            f'"{obj_a.description}" and "{obj_b.description}".'
        ),
        answer=f"{dist_m:.2f}",
        question_type=QuestionType.INTER_OBJECT_DISTANCE,
        task_group=SpatialTaskGroup.OBJECT_CENTRIC,
        answer_format=AnswerFormat.NUMERIC,
        num_views=2,
    )


def format_cardinal_cross_view(
    obj_a: ObjectInstance,
    obj_b: ObjectInstance,
    *,
    assumed_dir: str = "north",
    answer: str = "south",
) -> QAPair:
    return QAPair(
        question=(
            f'The direction of "{obj_a.description}" relative to image A is {assumed_dir}. '
            f'What is the direction of "{obj_b.description}" relative to image B?'
        ),
        answer=answer,
        question_type=QuestionType.CROSS_VIEW_CARDINAL,
        task_group=SpatialTaskGroup.CAMERA_OBJECT,
        answer_format=AnswerFormat.MCQ,
        options=["North", "South", "East", "West"],
        num_views=2,
    )


def euclidean_distance(a: ObjectInstance, b: ObjectInstance) -> float:
    ca, cb = a.center, b.center
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb, strict=True)))


def generate_qa_for_scene(scene: SceneAnnotation) -> list[QAPair]:
    """Instantiate a small QA set from 3D annotations (Sec. 3.1)."""
    out: list[QAPair] = [format_camera_translation_mcq()]
    labels = [o.label for o in scene.objects]
    out.append(format_object_counting("monitor", sum(1 for l in labels if l == "monitor")))
    if len(scene.objects) >= 2:
        d = euclidean_distance(scene.objects[0], scene.objects[1])
        out.append(format_inter_object_distance(scene.objects[0], scene.objects[1], d))
    if len(scene.objects) >= 2:
        out.append(
            format_cardinal_cross_view(scene.objects[0], scene.objects[1], assumed_dir="north", answer="east")
        )
    for q in out:
        q.scene_id = scene.scene_id
    return out
