"""OMAF fast multirate 360° encoding (Premkumar & Herglotz, arXiv:2601.17568)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2601.17568"
PAPER_DOI = "10.1145/3789239.3793270"
PAPER_TITLE = "Fast Multirate Encoding for 360° Video in OMAF Streaming Workflows"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

ENCODER = "x265"
DECODER_REF = "HM"
SJTU_SEQUENCES = 15
SJTU_ERP_SIZE = (4096, 8192)  # H×W (8192×4096 in paper: W×H)
CMP_FACE_SIZE = (2048, 2048)
CMP_FACES = 6

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "HD": (1024, 2048),
    "4K": (2048, 4096),
    "8K": (4096, 8192),
}

QPS = (22, 27, 32, 37, 42)
X265_REUSE_LEVEL = 10
X265_SCALE_FACTOR = 2  # HD→4K, 4K→8K


@dataclass
class OmaFmeConfig:
    variant: str = "ERP-CRC"
    anchor_quality: str = "HQ"  # LQ | MQ | HQ
    projection: str = "ERP"  # ERP | CMP
    parallel_faces: bool = False
