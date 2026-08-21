"""Optional live WebShop adapter."""

from __future__ import annotations

import os
from typing import Any


def webshop_importable() -> bool:
    try:
        import webshop  # noqa: F401

        return True
    except ImportError:
        return False


def probe_webshop() -> dict[str, Any]:
    data = os.environ.get("WEBSHOP_DATA", "")
    return {
        "package": webshop_importable(),
        "data": bool(data and os.path.isdir(data)),
        "ready": webshop_importable(),
        "install": "pip install webshop; clone https://github.com/princeton-nlp/WebShop",
        "env_var": "WEBSHOP_DATA",
    }
