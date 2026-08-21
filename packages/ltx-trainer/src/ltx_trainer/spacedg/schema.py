"""SpaceDG dataset / benchmark schema (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DegradationType(str, Enum):
    ORIGINAL = "original"
    DEFOCUS = "defocus"
    DISTORTION = "distortion"
    HAZE = "haze"
    JPEG_COMPRESSION = "jpeg_compression"
    LOW_LIGHT = "low_light"
    LOW_RESOLUTION = "low_resolution"
    MOTION_BLUR = "motion_blur"
    OVER_EXPOSURE = "over_exposure"
    WATER_DROPLETS = "water_droplets"


class SpatialTaskGroup(str, Enum):
    CAMERA_CENTRIC = "camera_centric"
    OBJECT_CENTRIC = "object_centric"
    CAMERA_OBJECT = "camera_object"


class QuestionType(str, Enum):
    CAMERA_TRANSLATION = "camera_translation"
    CAMERA_ROTATION = "camera_rotation"
    CAMERA_OBJECT_DISTANCE = "camera_object_distance_estimation"
    CAMERA_OBJECT_RELATIVE_DIRECTION = "camera_object_relative_direction"
    CROSS_VIEW_CARDINAL = "cross_view_cardinal_direction"
    INTER_OBJECT_DISTANCE = "inter_object_distance"
    OBJECT_PROXY_CARDINAL = "object_proxy_cardinal_direction"
    OBJECT_SIZE_COMPARISON = "object_size_comparison"
    OBJECT_BOUNDING_SIZE = "object_bounding_size_estimation"
    OBJECT_EXISTENCE = "object_existence_estimation"
    OBJECT_COUNTING = "object_counting"


class AnswerFormat(str, Enum):
    MCQ = "mcq"
    BINARY = "binary"
    NUMERIC = "numeric"
    LIST_NUMERIC = "list_numeric"


@dataclass
class QAPair:
    question: str
    answer: str
    question_type: QuestionType
    task_group: SpatialTaskGroup
    answer_format: AnswerFormat
    options: list[str] = field(default_factory=list)
    scene_id: str = ""
    num_views: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkItem:
    qa: QAPair
    image_paths: list[str]
    degradation: DegradationType
    param_range: str = ""
    clean_image_paths: list[str] = field(default_factory=list)
