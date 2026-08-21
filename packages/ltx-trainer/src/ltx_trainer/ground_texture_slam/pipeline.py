"""Multi-session SLAM pipeline stub (Sec. IV-A)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ltx_trainer.ground_texture_slam.kld import build_baseline_from_session
from ltx_trainer.ground_texture_slam.loop_closure import LoopCandidate, evaluate_loop_candidate
from ltx_trainer.ground_texture_slam.metrics import orientation_rmse_deg, position_rmse
from ltx_trainer.ground_texture_slam.overlap import visual_overlap
from ltx_trainer.ground_texture_slam.schema import LoopClosureMethod, Pose2D


@dataclass
class SLAMConfig:
    method: LoopClosureMethod = LoopClosureMethod.KLD
    loop_search_window: int = 5
    grayscale_kld: bool = False


@dataclass
class SessionResult:
    estimated: list[Pose2D]
    ground_truth: list[Pose2D]
    position_rmse: float
    orientation_rmse_deg: float
    loop_stats: dict[str, int]


@dataclass
class GroundTextureSLAM:
    cfg: SLAMConfig = field(default_factory=SLAMConfig)
    baseline: Tensor | None = None
    session0_images: list[Tensor] = field(default_factory=list)

    def set_baseline(self, images: list[Tensor]) -> None:
        self.session0_images = images
        self.baseline = build_baseline_from_session(
            images, grayscale=self.cfg.method == LoopClosureMethod.KLD_GRAY
        )

    def _estimate_poses_vo(self, gt: list[Pose2D]) -> list[Pose2D]:
        """Visual odometry stub: GT + small drift."""
        est: list[Pose2D] = []
        drift = 0.02 * (gt[0].session if gt else 0)
        for p in gt:
            est.append(Pose2D(p.x + drift, p.y + drift * 0.5, p.yaw + drift * 0.1, p.session, p.t))
        return est

    def process_session(
        self,
        images: list[Tensor],
        gt_poses: list[Pose2D],
    ) -> SessionResult:
        est = self._estimate_poses_vo(gt_poses)
        tp = fp = fn = tn = 0
        for i, (img, pose) in enumerate(zip(images, gt_poses)):
            for j in range(max(0, i - self.cfg.loop_search_window), i):
                cand_pose = gt_poses[j]
                true_loop = visual_overlap(pose, cand_pose)
                cand = LoopCandidate(
                    pose_a=pose,
                    pose_b=cand_pose,
                    image_a=images[i],
                    image_b=images[j],
                    is_true_loop=true_loop,
                )
                accept, _ = evaluate_loop_candidate(
                    cand, method=self.cfg.method, baseline=self.baseline
                )
                cand.accepted = accept
                if true_loop and accept:
                    tp += 1
                    # simple loop correction: pull estimate toward GT
                    est[i] = Pose2D(
                        0.7 * est[i].x + 0.3 * gt_poses[i].x,
                        0.7 * est[i].y + 0.3 * gt_poses[i].y,
                        est[i].yaw,
                        pose.session,
                        pose.t,
                    )
                elif true_loop and not accept:
                    fn += 1
                elif not true_loop and accept:
                    fp += 1
                    if self.cfg.method == LoopClosureMethod.ORIGINAL_SINGLE:
                        est[i] = Pose2D(est[i].x + 0.15, est[i].y - 0.1, est[i].yaw, pose.session, pose.t)
                else:
                    tn += 1
        return SessionResult(
            est,
            gt_poses,
            position_rmse(est, gt_poses),
            orientation_rmse_deg(est, gt_poses),
            {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        )


def evaluate_sessions(
    sessions: list[tuple[list[Tensor], list[Pose2D]]],
    *,
    method: LoopClosureMethod = LoopClosureMethod.KLD,
) -> dict[str, float | dict[str, int]]:
    slam = GroundTextureSLAM(SLAMConfig(method=method))
    if sessions:
        slam.set_baseline(sessions[0][0])
    all_est: list[Pose2D] = []
    all_gt: list[Pose2D] = []
    stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for images, gt in sessions:
        if method == LoopClosureMethod.ORIGINAL_MANY:
            slam = GroundTextureSLAM(SLAMConfig(method=method))
        res = slam.process_session(images, gt)
        all_est.extend(res.estimated)
        all_gt.extend(res.ground_truth)
        for k in stats:
            stats[k] += res.loop_stats[k]
    return {
        "position_rmse": position_rmse(all_est, all_gt),
        "orientation_rmse_deg": orientation_rmse_deg(all_est, all_gt),
        "loop_stats": stats,
    }
