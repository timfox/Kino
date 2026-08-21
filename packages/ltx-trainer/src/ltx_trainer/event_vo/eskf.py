"""Error-State Kalman Filter monocular VO (Sec. III)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ltx_trainer.event_vo.config import EventVOConfig
from ltx_trainer.event_vo.lie import rot_from_theta
from ltx_trainer.event_vo.projection import project_with_jacobians
from ltx_trainer.event_vo.rate import FeatureUpdate
from ltx_trainer.event_vo.triangulation import triangulate_dlt


@dataclass
class _PendingFeature:
  first_uv: np.ndarray
  clone_p: np.ndarray
  clone_theta: np.ndarray
  last_uv: np.ndarray


@dataclass
class MonocularEventESKF:
    cfg: EventVOConfig = field(default_factory=EventVOConfig)
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    landmarks: dict[int, np.ndarray] = field(default_factory=dict)
    landmark_order: list[int] = field(default_factory=list)
    _pending: dict[int, _PendingFeature] = field(default_factory=dict)
    P: np.ndarray | None = None
    initialized: bool = False
    last_time: float = 0.0
    trajectory: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def _state_dim(self) -> int:
        return 9 + 3 * len(self.landmark_order)

    def _init_covariance(self) -> None:
        n = self._state_dim()
        self.P = np.eye(n, dtype=np.float64) * 0.01

    def _landmark_index(self, fid: int) -> int:
        return self.landmark_order.index(fid)

    def propagate(self, t: float) -> None:
        dt = max(t - self.last_time, 0.0)
        if dt <= 0:
            return
        self.p = self.p + self.v * dt
        self.last_time = t
        if self.P is None:
            return
        n = self._state_dim()
        F = np.eye(n)
        F[0:3, 6:9] = np.eye(3) * dt
        q = self.cfg.process_noise_pos
        Q = np.eye(n) * 1e-6
        Q[0:3, 0:3] = np.eye(3) * (q * q * dt)
        Q[3:6, 3:6] = np.eye(3) * (self.cfg.process_noise_rot**2 * dt)
        Q[6:9, 6:9] = np.eye(3) * (self.cfg.process_noise_vel**2 * dt)
        self.P = F @ self.P @ F.T + Q

    def _marginalize_landmark(self, fid: int) -> None:
        if fid not in self.landmark_order:
            return
        idx = self._landmark_index(fid)
        base = 9 + 3 * idx
        keep = [i for i in range(self._state_dim()) if not (base <= i < base + 3)]
        self.P = self.P[np.ix_(keep, keep)]
        del self.landmarks[fid]
        self.landmark_order.pop(idx)
        self._pending.pop(fid, None)

    def _augment_landmark(self, fid: int, p_w: np.ndarray, Gx: np.ndarray | None = None) -> None:
        if fid in self.landmarks:
            return
        if len(self.landmark_order) >= self.cfg.max_landmarks:
            self._marginalize_landmark(self.landmark_order[0])
        self.landmarks[fid] = p_w.copy()
        self.landmark_order.append(fid)
        n_old = self._state_dim() - 3
        P_new = np.zeros((n_old + 3, n_old + 3))
        if self.P is not None:
            P_new[:n_old, :n_old] = self.P
        if Gx is not None and self.P is not None:
            P_new[:n_old, n_old:] = self.P @ Gx.T
            P_new[n_old:, :n_old] = Gx @ self.P
            Rm = np.eye(3) * (self.cfg.measurement_noise_px**2)
            P_new[n_old:, n_old:] = Gx @ self.P @ Gx.T + Rm
        else:
            P_new[n_old:, n_old:] = np.eye(3) * 0.05
        self.P = P_new
        self._pending.pop(fid, None)

    def _try_triangulate(self, fid: int) -> bool:
        pend = self._pending.get(fid)
        if pend is None:
            return False
        parallax = np.linalg.norm(pend.last_uv - pend.first_uv)
        if parallax < self.cfg.min_parallax_px:
            return False
        pt = triangulate_dlt(
            pend.first_uv,
            pend.last_uv,
            pend.clone_p,
            pend.clone_theta,
            self.p,
            self.theta,
            self.cfg.fx,
            self.cfg.fy,
            self.cfg.cx,
            self.cfg.cy,
        )
        if pt is None:
            return False
        self._augment_landmark(fid, pt)
        return True

    def correct(self, upd: FeatureUpdate) -> None:
        fid = upd.feature_id
        z = np.array([upd.u, upd.v], dtype=np.float64)

        if fid in self.landmarks:
            p_w = self.landmarks[fid]
            z_hat, Hp, Hr, Hl = project_with_jacobians(
                p_w, self.p, self.theta, self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy
            )
            innov = z - z_hat
            n = self._state_dim()
            H = np.zeros((2, n))
            H[:, 0:3] = Hp
            H[:, 3:6] = Hr
            li = self._landmark_index(fid)
            H[:, 9 + 3 * li : 9 + 3 * (li + 1)] = Hl
            if self.P is not None:
                Rm = np.eye(2) * (self.cfg.measurement_noise_px**2)
                S = H @ self.P @ H.T + Rm + np.eye(2) * 1e-4
                try:
                    dx = self.P @ H.T @ np.linalg.solve(S, innov)
                except np.linalg.LinAlgError:
                    return
                self.p += dx[0:3]
                self.theta += dx[3:6]
                self.v += dx[6:9]
                for i, lid in enumerate(self.landmark_order):
                    self.landmarks[lid] += dx[9 + 3 * i : 9 + 3 * (i + 1)]
                K = np.linalg.solve(S.T, H @ self.P).T
                self.P = (np.eye(n) - K @ H) @ self.P
                self.P = 0.5 * (self.P + self.P.T)
            return

        if fid in self._pending:
            self._pending[fid].last_uv = z
            self._try_triangulate(fid)
            return

        self._pending[fid] = _PendingFeature(
            first_uv=z.copy(),
            clone_p=self.p.copy(),
            clone_theta=self.theta.copy(),
            last_uv=z.copy(),
        )

    def process_update(self, upd: FeatureUpdate) -> None:
        self.propagate(upd.time_s)
        if not self.initialized:
            return
        self.correct(upd)
        self.trajectory.append((upd.time_s, self.p.copy()))

    def mark_initialized(self) -> None:
        self.initialized = True
        self._init_covariance()

    def delete_features(self, ids: list[int]) -> None:
        for fid in ids:
            self._marginalize_landmark(fid)
