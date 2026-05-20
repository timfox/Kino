"""Structure-from-motion export for generated (or any) videos.

Extracts frames, runs reconstruction, and writes agent-friendly artifacts:

- ``sfm_summary.json`` — metadata, backend, stats
- ``cameras.json`` — per-frame intrinsics and world-to-camera matrices
- ``trajectories.npy`` — ``[N, 4, 4]`` float32 world-to-camera
- ``sparse.ply`` — sparse point cloud (ASCII PLY)
- ``frames/`` — extracted JPEGs used for reconstruction
- ``colmap/`` — full COLMAP model when the CLI backend succeeds

Backends (``SFM_BACKEND`` env or ``backend=`` argument):

- ``auto`` — try COLMAP CLI, else OpenCV incremental
- ``colmap`` — require COLMAP on ``PATH``
- ``opencv`` — ORB + essential-matrix chain (no external binaries)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

SfmBackend = Literal["auto", "colmap", "opencv"]


class SfmExportError(RuntimeError):
    pass


@dataclass
class CameraFrame:
    frame_index: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: list[list[float]]  # 4x4 row-major


@dataclass
class SfmResult:
    video_path: str
    output_dir: str
    backend: str
    num_frames: int
    num_points: int
    mean_reproj_error: float | None
    cameras: list[CameraFrame] = field(default_factory=list)
    points_xyz: list[list[float]] = field(default_factory=list)
    points_rgb: list[list[int]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "backend": self.backend,
            "num_frames": self.num_frames,
            "num_points": self.num_points,
            "mean_reproj_error": self.mean_reproj_error,
            "num_cameras": len(self.cameras),
        }


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def resolve_backend(backend: SfmBackend | None) -> str:
    b = (backend or os.environ.get("SFM_BACKEND", "auto")).strip().lower()
    if b not in ("auto", "colmap", "opencv"):
        raise SfmExportError(f"Unknown SFM backend {b!r}")
    if b == "auto":
        return "colmap" if colmap_available() else "opencv"
    if b == "colmap" and not colmap_available():
        raise SfmExportError("SFM backend 'colmap' requested but colmap is not on PATH")
    return b


def _default_intrinsics(width: int, height: int, fov_deg: float = 55.0) -> tuple[float, float, float, float]:
    fov = np.deg2rad(fov_deg)
    fx = 0.5 * width / np.tan(0.5 * fov)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    max_frames: int = 120,
    target_fps: float | None = 2.0,
    jpeg_quality: int = 95,
) -> list[Path]:
    """Write ``frame_%06d.jpg`` under ``frames_dir``; return sorted paths."""
    from ltx_trainer.ffmpeg_io import extract_frame_jpeg, probe_media, sample_frame_times

    frames_dir.mkdir(parents=True, exist_ok=True)
    pr = probe_media(video_path)
    duration = pr.duration_sec
    n = max(8, min(max_frames, int(duration * (target_fps or 2.0)) + 1))
    times = sample_frame_times(duration, n)
    paths: list[Path] = []
    for i, t in enumerate(times):
        out = frames_dir / f"frame_{i:06d}.jpg"
        if not extract_frame_jpeg(video_path, t, out):
            continue
        paths.append(out)
    if len(paths) < 2:
        raise SfmExportError(f"Need at least 2 extracted frames from {video_path}, got {len(paths)}")
    return paths


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    lines = ["ply", "format ascii 1.0", f"element vertex {n}", "property float x", "property float y", "property float z"]
    if rgb is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        for i in range(n):
            x, y, z = xyz[i]
            if rgb is not None:
                r, g, b = rgb[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _parse_colmap_images_txt(path: Path) -> dict[str, tuple[int, np.ndarray]]:
    """Parse COLMAP ``images.txt`` → image name → (camera_id, 4x4 w2c)."""
    poses: dict[str, tuple[int, np.ndarray]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        R = _quat_wxyz_to_rot(q)
        t = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3:4] = t
        poses[name] = (cam_id, w2c)
        if i < len(lines) and not lines[i].strip().startswith("#"):
            i += 1
    return poses


def _parse_colmap_cameras_txt(path: Path) -> dict[int, tuple[int, int, float, float, float, float]]:
    cams: dict[int, tuple[int, int, float, float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        cid = int(p[0])
        model = p[1]
        w, h = int(p[2]), int(p[3])
        params = list(map(float, p[4:]))
        if model in ("PINHOLE", "OPENCV") and len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE" and len(params) >= 3:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx, fy, cx, cy = _default_intrinsics(w, h)
        cams[cid] = (w, h, fx, fy, cx, cy)
    return cams


def _parse_colmap_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz_list: list[list[float]] = []
    rgb_list: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        xyz_list.append([float(p[1]), float(p[2]), float(p[3])])
        rgb_list.append([int(p[4]), int(p[5]), int(p[6])])
    if not xyz_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.array(xyz_list, dtype=np.float64), np.array(rgb_list, dtype=np.uint8)


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _run_colmap(workspace: Path, frame_paths: list[Path]) -> Path:
    """Run COLMAP sparse pipeline; return sparse model directory."""
    colmap = shutil.which("colmap")
    if not colmap:
        raise SfmExportError("colmap not on PATH")

    images = workspace / "images"
    images.mkdir(parents=True, exist_ok=True)
    for p in frame_paths:
        shutil.copy2(p, images / p.name)

    db = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    def run(args: list[str]) -> None:
        cmd = [colmap, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SfmExportError(
                f"colmap failed: {' '.join(args)}",
            )

    if db.exists():
        db.unlink()
    run(["feature_extractor", "--database_path", str(db), "--image_path", str(images)])
    run(["exhaustive_matcher", "--database_path", str(db)])
    run(["mapper", "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse)])

    models = sorted(sparse.glob("*"))
    if not models:
        raise SfmExportError("COLMAP mapper produced no model")
    best = models[0]
    txt_dir = workspace / "sparse_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run(["model_converter", "--input_path", str(best), "--output_path", str(txt_dir), "--output_type", "TXT"])
    return txt_dir


def _reconstruct_colmap(frame_paths: list[Path], workspace: Path) -> SfmResult:
    txt = _run_colmap(workspace, frame_paths)
    cams = _parse_colmap_cameras_txt(txt / "cameras.txt")
    poses = _parse_colmap_images_txt(txt / "images.txt")
    xyz, rgb = _parse_colmap_points3d_txt(txt / "points3D.txt")

    cameras: list[CameraFrame] = []
    for i, fp in enumerate(frame_paths):
        name = fp.name
        cam_id, w2c = poses.get(name, (1, np.eye(4, dtype=np.float64)))
        w, h, fx, fy, cx, cy = cams.get(cam_id, (640, 480, *_default_intrinsics(640, 480)))
        cameras.append(
            CameraFrame(
                frame_index=i,
                image_name=name,
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                world_to_camera=w2c.tolist(),
            )
        )

    return SfmResult(
        video_path="",
        output_dir=str(workspace),
        backend="colmap",
        num_frames=len(frame_paths),
        num_points=int(xyz.shape[0]),
        mean_reproj_error=None,
        cameras=cameras,
        points_xyz=xyz.tolist(),
        points_rgb=rgb.tolist(),
    )


def _reconstruct_opencv(frame_paths: list[Path], width: int, height: int) -> SfmResult:
    import cv2

    fx, fy, cx, cy = _default_intrinsics(width, height)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    orb = cv2.ORB_create(nfeatures=4000)
    grays: list[np.ndarray] = []
    kps: list[list[cv2.KeyPoint]] = []
    descs: list[np.ndarray] = []
    for fp in frame_paths:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SfmExportError(f"Failed to read {fp}")
        grays.append(img)
        k, d = orb.detectAndCompute(img, None)
        kps.append(k or [])
        descs.append(d if d is not None else np.zeros((0, 32), dtype=np.uint8))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    for i in range(len(frame_paths) - 1):
        if len(descs[i]) == 0 or len(descs[i + 1]) == 0:
            poses.append(poses[-1].copy())
            continue
        matches = bf.knnMatch(descs[i], descs[i + 1], k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < 8:
            poses.append(poses[-1].copy())
            continue
        pts1 = np.float32([kps[i][m.queryIdx].pt for m in good])
        pts2 = np.float32([kps[i + 1][m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(poses[-1].copy())
            continue
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.ravel()
        T_next = poses[-1] @ np.linalg.inv(T_rel)
        poses.append(T_next)

    # pad if needed
    while len(poses) < len(frame_paths):
        poses.append(poses[-1].copy())

    # triangulate from frame 0 and each i
    P0 = K @ poses[0][:3, :]
    for i in range(1, len(frame_paths)):
        if len(descs[0]) == 0 or len(descs[i]) == 0:
            continue
        matches = bf.knnMatch(descs[0], descs[i], k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < 8:
            continue
        pts0 = np.float32([kps[0][m.queryIdx].pt for m in good])
        ptsi = np.float32([kps[i][m.trainIdx].pt for m in good])
        Pi = K @ poses[i][:3, :]
        X_h = cv2.triangulatePoints(P0, Pi, pts0.T, ptsi.T)
        X = (X_h[:3] / X_h[3:4]).T
        mask_z = X[:, 2] > 0.01
        X = X[mask_z]
        if X.size == 0:
            continue
        bgr0 = cv2.imread(str(frame_paths[0]))
        h0, w0 = bgr0.shape[:2]
        colors = np.array(
            [bgr0[min(int(p[1]), h0 - 1), min(int(p[0]), w0 - 1)] for p in pts0[mask_z]],
            dtype=np.uint8,
        )
        colors = colors[:, ::-1]
        all_xyz.append(X)
        all_rgb.append(colors)

    if all_xyz:
        xyz = np.vstack(all_xyz)
        rgb = np.vstack(all_rgb).astype(np.uint8)
    else:
        xyz = np.zeros((0, 3))
        rgb = np.zeros((0, 3), dtype=np.uint8)

    cameras: list[CameraFrame] = []
    for i, fp in enumerate(frame_paths):
        w2c = np.linalg.inv(poses[i])
        cameras.append(
            CameraFrame(
                frame_index=i,
                image_name=fp.name,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                world_to_camera=w2c.tolist(),
            )
        )

    return SfmResult(
        video_path="",
        output_dir="",
        backend="opencv",
        num_frames=len(frame_paths),
        num_points=int(xyz.shape[0]),
        mean_reproj_error=None,
        cameras=cameras,
        points_xyz=xyz.tolist(),
        points_rgb=rgb.tolist(),
    )


def export_sfm_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    backend: SfmBackend | None = None,
    max_frames: int = 120,
    target_fps: float | None = 2.0,
    fov_deg: float = 55.0,
) -> SfmResult:
    """Run SfM on ``video_path`` and write artifacts under ``output_dir``."""
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frame_paths = extract_frames(video_path, frames_dir, max_frames=max_frames, target_fps=target_fps)

    import cv2

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise SfmExportError(f"Cannot read frame {frame_paths[0]}")
    height, width = sample.shape[:2]

    resolved = resolve_backend(backend)
    workspace = output_dir / "colmap_workspace"
    if resolved == "colmap":
        try:
            result = _reconstruct_colmap(frame_paths, workspace)
        except SfmExportError:
            if backend == "colmap":
                raise
            result = _reconstruct_opencv(frame_paths, width, height)
            result.backend = "opencv_fallback"
    else:
        result = _reconstruct_opencv(frame_paths, width, height)

    result.video_path = str(video_path)
    result.output_dir = str(output_dir)

    # persist
    traj = np.stack(
        [np.array(c.world_to_camera, dtype=np.float32) for c in result.cameras],
        axis=0,
    )
    np.save(output_dir / "trajectories.npy", traj)

    with (output_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in result.cameras], f, indent=2)

    summary = result.to_summary_dict()
    summary["fov_deg_assumed"] = fov_deg
    summary["target_fps"] = target_fps
    with (output_dir / "sfm_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if result.points_xyz:
        xyz = np.array(result.points_xyz, dtype=np.float64)
        rgb = np.array(result.points_rgb, dtype=np.uint8) if result.points_rgb else None
        _write_ply(output_dir / "sparse.ply", xyz, rgb)

    return result
