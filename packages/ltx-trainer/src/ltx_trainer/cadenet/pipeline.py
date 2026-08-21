"""Three-thread CADENet orchestration (Sec. III-B)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ltx_trainer.cadenet.cape import enhance
from ltx_trainer.cadenet.detector import YOLOStub
from ltx_trainer.cadenet.eg_nms import score_detections
from ltx_trainer.cadenet.ktt import KTTConfig, predict_tracks, update_tracks
from ltx_trainer.cadenet.pee import patch_entropy_reliability
from ltx_trainer.cadenet.schema import Detection, Track, WeatherCondition
from ltx_trainer.cadenet.thread_e import CLIPWeatherStub
from ltx_trainer.cadenet.wem import classify_weather


@dataclass
class CADENetConfig:
    conf_thresh: float = 0.25
    kalman_lag_frames: int = 3
    clip_ambiguity: float = 0.15
    thread_s_fps_target: float = 44.0


@dataclass
class FrameResult:
    tracks: list[Track]
    detections_s: list[Detection]
    detections_fused: list[Detection]
    weather: WeatherCondition
    severity: float


@dataclass
class CADENet:
    cfg: CADENetConfig = field(default_factory=CADENetConfig)
    detector_s: YOLOStub = field(default_factory=lambda: YOLOStub(fast=True))
    detector_q: YOLOStub = field(default_factory=lambda: YOLOStub(fast=False))
    clip: CLIPWeatherStub = field(default_factory=CLIPWeatherStub)
    tracks: list[Track] = field(default_factory=list)
    next_track_id: int = 0
    clip_slot: WeatherCondition | None = None

    def thread_e_update(self, frame: Tensor) -> None:
        """Background CLIP classification (lock-free slot)."""
        self.clip_slot = self.clip.classify(frame)

    def process_frame(self, frame: Tensor) -> FrameResult:
        """Synchronous demo of S + Q fusion; E slot read at CAPE entry."""
        h, w = frame.shape[-2:]
        # Thread S — safety path
        dets_s = self.detector_s.detect(frame)
        self.tracks, self.next_track_id = update_tracks(
            self.tracks, dets_s, next_id=self.next_track_id
        )
        # Thread E (would run async; here we refresh slot)
        self.thread_e_update(frame)
        wem = classify_weather(frame, clip_label=self.clip_slot, clip_threshold=self.cfg.clip_ambiguity)
        r_map = patch_entropy_reliability(frame.mean(dim=0))
        enhanced = enhance(frame, wem.condition, severity=wem.severity)
        dets_q = self.detector_q.detect(enhanced)
        projected = predict_tracks(
            [Track(0, d, d.conf, kalman_state=[0, 0, 1, 1, 0, 0, 0]) for d in dets_q],
            steps=self.cfg.kalman_lag_frames,
        )
        dets_q_proj = [t.box for t in projected]
        fused = score_detections(dets_s, dets_q_proj, r_map, img_h=h, img_w=w)
        self.tracks, self.next_track_id = update_tracks(
            self.tracks, fused, next_id=self.next_track_id
        )
        return FrameResult(self.tracks, dets_s, fused, wem.condition, wem.severity)


def process_frame(model: CADENet, frame: Tensor) -> FrameResult:
    return model.process_frame(frame)
