"""sRGB / Oklab / Oklch conversions (Ottosson Oklab)."""

from __future__ import annotations

import numpy as np

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 0.0) ** (1.0 / 2.4)) - 0.055)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = rgb @ _M1.T
    lms = np.maximum(lms, 0.0)
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lms = lms_**3
    return lms @ _M1_INV.T


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_oklch(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = np.asarray(lab, dtype=np.float64)
    l_ = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    c = np.hypot(a, b)
    h = np.arctan2(b, a)
    return l_, c, h


def oklch_to_oklab(l_: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    a = c * np.cos(h)
    b = c * np.sin(h)
    return np.stack([l_, a, b], axis=-1)
