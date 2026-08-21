"""Structure-from-motion export from video or image folders (OpenCV / optional COLMAP)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

SfmBackendName = Literal["auto", "colmap", "opencv"]


@dataclass
class SfmCamera:
    frame_index: int
    image_name: str
    width: int
    height: int
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SfmExportResult:
    backend: str
    num_frames: int
    num_points: int
    cameras: list[SfmCamera] = field(default_factory=list)
    video_path: str | None = None
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "num_frames": self.num_frames,
            "num_points": self.num_points,
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "cameras": [c.to_dict() for c in self.cameras],
        }


def resolve_backend(name: SfmBackendName) -> Literal["colmap", "opencv"]:
    if name == "opencv":
        return "opencv"
    if name == "colmap":
        return "colmap"
    return "colmap" if shutil.which("colmap") else "opencv"


def _extract_frames_ffmpeg(
    video: Path,
    frames_dir: Path,
    *,
    target_fps: float,
    max_frames: int,
) -> list[Path]:
    from ltx_trainer.ffmpeg_io import ffmpeg_bin, require_ffmpeg

    frames_dir.mkdir(parents=True, exist_ok=True)
    require_ffmpeg()
    pattern = str(frames_dir / "frame_%06d.jpg")
    cmd = [
        ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={target_fps}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 frames from {video}, got {len(frames)}")
    return frames


def _camera_matrix(w: int, h: int, fov_deg: float = 70.0) -> np.ndarray:
    fov = np.deg2rad(fov_deg)
    fx = 0.5 * w / np.tan(fov * 0.5)
    fy = fx
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _export_sfm_opencv(
    image_paths: list[Path],
    output_dir: Path,
    *,
    source_video: str | None,
    pano360_meta: dict[str, Any] | None,
) -> SfmExportResult:
    cv2 = __import__("cv2")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for i, src in enumerate(image_paths):
        dst = frames_dir / f"frame_{i:06d}.jpg"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        staged.append(dst)

    orb = cv2.ORB_create(2000)
    all_kp: list[Any] = []
    all_des: list[Any] = []
    sizes: list[tuple[int, int]] = []
    for p in staged:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Unreadable image: {p}")
        sizes.append((gray.shape[1], gray.shape[0]))
        kp, des = orb.detectAndCompute(gray, None)
        all_kp.append(kp)
        all_des.append(des)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    points3d: list[np.ndarray] = []
    K0 = _camera_matrix(sizes[0][0], sizes[0][1])

    for i in range(1, len(staged)):
        if all_des[i] is None or all_des[i - 1] is None:
            poses.append(poses[-1].copy())
            continue
        matches = bf.knnMatch(all_des[i - 1], all_des[i], k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 8:
            poses.append(poses[-1].copy())
            continue
        pts0 = np.float32([all_kp[i - 1][m.queryIdx].pt for m in good])
        pts1 = np.float32([all_kp[i][m.trainIdx].pt for m in good])
        Ki = _camera_matrix(sizes[i][0], sizes[i][1])
        E, _ = cv2.findEssentialMat(pts0, pts1, K0, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(poses[-1].copy())
            continue
        _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K0)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        poses.append(poses[-1] @ T)
        P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = Ki @ np.hstack([R, t])
        X = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
        X = (X[:3] / X[3:4]).T
        points3d.append(X[np.isfinite(X).all(axis=1)])

    if points3d:
        pts = np.concatenate(points3d, axis=0)
    else:
        pts = np.zeros((0, 3), dtype=np.float64)

    cameras: list[SfmCamera] = []
    traj = np.zeros((len(staged), 4, 4), dtype=np.float64)
    for i, p in enumerate(staged):
        w, h = sizes[i]
        Ki = _camera_matrix(w, h)
        R = poses[i][:3, :3]
        t = poses[i][:3, 3]
        cameras.append(
            SfmCamera(
                frame_index=i,
                image_name=p.name,
                width=w,
                height=h,
                K=Ki.tolist(),
                R=R.tolist(),
                t=t.tolist(),
            )
        )
        traj[i] = poses[i]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cameras.json").write_text(
        json.dumps([c.to_dict() for c in cameras], indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "trajectories.npy", traj)
    _write_sparse_ply(output_dir / "sparse.ply", pts)
    summary = {
        "backend": "opencv",
        "num_frames": len(cameras),
        "num_points": int(pts.shape[0]),
        "video_path": source_video,
        "output_dir": str(output_dir),
    }
    if pano360_meta:
        summary["pano360"] = pano360_meta
    (output_dir / "sfm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return SfmExportResult(
        backend="opencv",
        num_frames=len(cameras),
        num_points=int(pts.shape[0]),
        cameras=cameras,
        video_path=source_video,
        output_dir=str(output_dir),
    )


def _write_sparse_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 180 120 80\n")


def _export_sfm_colmap(
    image_paths: list[Path],
    output_dir: Path,
    *,
    source_video: str | None,
    pano360_meta: dict[str, Any] | None,
) -> SfmExportResult:
    workspace = output_dir / "colmap_workspace"
    images = workspace / "images"
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    if workspace.exists():
        shutil.rmtree(workspace)
    images.mkdir(parents=True)
    for i, src in enumerate(image_paths):
        shutil.copy2(src, images / f"{src.stem if src.suffix else f'img_{i:06d}'}{src.suffix or '.jpg'}")

    colmap = shutil.which("colmap")
    if not colmap:
        raise RuntimeError("colmap not on PATH")

    subprocess.run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
        ],
        check=True,
    )
    subprocess.run(
        [colmap, "exhaustive_matcher", "--database_path", str(database)],
        check=True,
    )
    sparse.mkdir(parents=True)
    subprocess.run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ],
        check=True,
    )

    # COLMAP sparse under workspace/; poses/point cloud via OpenCV triangulation for bundle compatibility.
    result = _export_sfm_opencv(image_paths, output_dir, source_video=source_video, pano360_meta=pano360_meta)
    result.backend = "colmap"
    summary = json.loads((output_dir / "sfm_summary.json").read_text(encoding="utf-8"))
    summary["backend"] = "colmap"
    summary["colmap_workspace"] = str(workspace)
    (output_dir / "sfm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return SfmExportResult(
        backend="colmap",
        num_frames=result.num_frames,
        num_points=result.num_points,
        cameras=result.cameras,
        video_path=source_video,
        output_dir=str(output_dir),
    )


def export_sfm_from_images(
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    backend: SfmBackendName = "auto",
    source_video: str | None = None,
    pano360_meta: dict[str, Any] | None = None,
    image_glob: str = "*.jpg",
) -> SfmExportResult:
    """Run SfM on a directory of perspective images."""
    images_dir = Path(images_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    paths = sorted(images_dir.glob(image_glob))
    if len(paths) < 2:
        paths = sorted(images_dir.glob("*.png"))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {images_dir}")

    resolved = resolve_backend(backend)
    if resolved == "colmap":
        return _export_sfm_colmap(paths, output_dir, source_video=source_video, pano360_meta=pano360_meta)
    return _export_sfm_opencv(paths, output_dir, source_video=source_video, pano360_meta=pano360_meta)


def export_sfm_from_video(
    video: str | Path,
    output_dir: str | Path,
    *,
    backend: SfmBackendName = "auto",
    max_frames: int = 120,
    target_fps: float = 2.0,
    projection: Literal["flat", "equirectangular", "auto"] = "auto",
    views_per_frame: int = 8,
) -> SfmExportResult:
    """Export SfM from a video file.

    When ``projection`` is ``equirectangular`` or ``auto`` detects 2:1 aspect,
    delegates to :func:`run_pano360_photogrammetry`.
    """
    video_path = Path(video).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    use_pano = projection == "equirectangular"
    if projection == "auto":
        from ltx_trainer.ffmpeg_io import ffprobe_json, require_ffmpeg

        try:
            require_ffmpeg()
            raw = ffprobe_json(video_path)
            for st in raw.get("streams") or []:
                if st.get("codec_type") == "video":
                    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
                    from ltx_trainer.pano360.geometry import is_equirectangular_size

                    use_pano = is_equirectangular_size(w, h)
                    break
        except Exception:
            use_pano = False

    if use_pano:
        from ltx_trainer.pano360.pipeline import Pano360PhotogrammetryConfig, run_pano360_photogrammetry

        pano_out = output_dir if output_dir.name != "sfm" else output_dir.parent
        result = run_pano360_photogrammetry(
            video_path,
            Pano360PhotogrammetryConfig(
                output_dir=pano_out,
                target_fps=target_fps,
                max_frames=max_frames,
                views_per_frame=views_per_frame,
                sfm_backend=backend,
            ),
        )
        summary_path = result.sfm_dir / "sfm_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        cameras_data = json.loads((result.sfm_dir / "cameras.json").read_text(encoding="utf-8"))
        cameras = [SfmCamera(**c) for c in cameras_data]
        return SfmExportResult(
            backend=result.sfm_backend,
            num_frames=result.num_equirect_frames,
            num_points=int(summary.get("num_points") or 0),
            cameras=cameras,
            video_path=str(video_path),
            output_dir=str(result.sfm_dir),
        )

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frame_paths = _extract_frames_ffmpeg(video_path, frames_dir, target_fps=target_fps, max_frames=max_frames)
    return export_sfm_from_images(
        frames_dir,
        output_dir,
        backend=backend,
        source_video=str(video_path),
        image_glob="frame_*.jpg",
    )
