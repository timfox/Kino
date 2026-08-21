"""LTX-2/2.3 legal frame counts (training + inference constraint)."""


def legal_frame_count(raw_frames: int) -> int:
    """Largest frame count <= *raw_frames* with ``frames % 8 == 1`` (LTX temporal VAE)."""
    if raw_frames < 1:
        return 1
    n = raw_frames
    while n > 1 and (n % 8) != 1:
        n -= 1
    return max(1, n)


def frames_for_duration(*, duration_s: float, fps: float) -> int:
    """Pixel frames for a wall-clock span, snapped to LTX legality."""
    raw = int(round(max(0.0, duration_s) * max(fps, 1e-6)))
    return legal_frame_count(max(1, raw))


def duration_for_frames(*, frames: int, fps: float) -> float:
    return float(frames) / max(float(fps), 1e-6)
