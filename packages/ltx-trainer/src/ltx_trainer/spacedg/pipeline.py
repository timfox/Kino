"""3DGS degradation data engine (Fig. 3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.spacedg.degradations import DegradationParams, apply_degradation, sample_params
from ltx_trainer.spacedg.qa_templates import generate_qa_for_scene
from ltx_trainer.spacedg.scene import SceneAnnotation, render_clean_view, synthetic_scannet_scene
from ltx_trainer.spacedg.schema import BenchmarkItem, DegradationType, QAPair


MIXTURE_RECIPES: dict[str, list[DegradationType]] = {
    "night_capture": [DegradationType.LOW_LIGHT, DegradationType.MOTION_BLUR, DegradationType.LOW_RESOLUTION],
    "hazy_long_range": [DegradationType.HAZE, DegradationType.LOW_RESOLUTION, DegradationType.MOTION_BLUR],
    "wet_lens_motion": [DegradationType.WATER_DROPLETS, DegradationType.MOTION_BLUR, DegradationType.LOW_LIGHT],
    "backlit_dynamic": [DegradationType.OVER_EXPOSURE, DegradationType.MOTION_BLUR],
    "motion_defocus": [DegradationType.MOTION_BLUR, DegradationType.DEFOCUS, DegradationType.LOW_RESOLUTION],
    "compressed_portrait": [DegradationType.DEFOCUS, DegradationType.JPEG_COMPRESSION],
}


@dataclass
class SpaceDGEngine:
    height: int = 256
    width: int = 256

    def render_degraded(
        self,
        scene: SceneAnnotation,
        degradation: DegradationType,
        *,
        view_idx: int = 0,
        params: DegradationParams | None = None,
    ) -> Tensor:
        rgb, depth = render_clean_view(scene, view_idx=view_idx, height=self.height, width=self.width)
        if degradation == DegradationType.ORIGINAL:
            return rgb
        return apply_degradation(rgb, degradation, depth=depth, params=params)

    def render_mixture(self, scene: SceneAnnotation, recipe: str, *, view_idx: int = 0) -> Tensor:
        types = MIXTURE_RECIPES.get(recipe, [DegradationType.HAZE])
        rgb, depth = render_clean_view(scene, view_idx=view_idx, height=self.height, width=self.width)
        for i, dt in enumerate(types):
            p = sample_params(dt) if i == 0 else sample_params(dt)
            rgb = apply_degradation(rgb, dt, depth=depth, params=p if i == 0 else None)
        return rgb

    def build_vqa_instances(
        self,
        qa: QAPair,
        scene: SceneAnnotation,
        *,
        out_dir: Path | None = None,
    ) -> list[BenchmarkItem]:
        from ltx_trainer.spacedg.degradations import DEGRADATION_PARAM_RANGES

        items: list[BenchmarkItem] = []
        degradations = [DegradationType.ORIGINAL] + [
            d for d in DegradationType if d != DegradationType.ORIGINAL
        ]
        for deg in degradations:
            paths: list[str] = []
            for v in range(qa.num_views):
                img = self.render_degraded(scene, deg, view_idx=v)
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    p = out_dir / f"{scene.scene_id}_{qa.question_type.value}_{deg.value}_v{v}.png"
                    _save_rgb(p, img)
                    paths.append(str(p))
                else:
                    paths.append(f"mem://{scene.scene_id}/{deg.value}/v{v}")
            pr = DEGRADATION_PARAM_RANGES.get(deg.value, "") if deg != DegradationType.ORIGINAL else ""
            items.append(BenchmarkItem(qa=qa, image_paths=paths, degradation=deg, param_range=pr))
        return items


def synthesize_benchmark_item(
    *,
    scene_id: str = "demo_001",
    out_dir: Path | str | None = None,
) -> list[BenchmarkItem]:
    scene = synthetic_scannet_scene(scene_id=scene_id)
    engine = SpaceDGEngine()
    od = Path(out_dir) if out_dir else None
    all_items: list[BenchmarkItem] = []
    for qa in generate_qa_for_scene(scene):
        all_items.extend(engine.build_vqa_instances(qa, scene, out_dir=od))
    return all_items


def _save_rgb(path: Path, rgb: Tensor) -> None:
    from PIL import Image

    arr = (rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)
