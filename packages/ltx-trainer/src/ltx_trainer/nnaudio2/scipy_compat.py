"""CFP SciPy import path fix (Sec. 6.3)."""

from __future__ import annotations

LEGACY_IMPORT = "scipy.signal.blackmanharris"
MODERN_IMPORT = "scipy.signal.windows.blackmanharris"


def blackmanharris_import_path(use_modern: bool = True) -> str:
    return MODERN_IMPORT if use_modern else LEGACY_IMPORT


def cfp_window_available(use_modern: bool = True) -> bool:
    try:
        if use_modern:
            from scipy.signal.windows import blackmanharris  # noqa: F401
        else:
            import scipy.signal as sig

            return hasattr(sig, "blackmanharris")
        return True
    except ImportError:
        return False
