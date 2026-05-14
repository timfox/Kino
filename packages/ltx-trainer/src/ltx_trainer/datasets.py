import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"
# Some preprocess scripts omit the leading dot; accept both.
_PRECOMPUTED_DIR_ALIASES: tuple[str, ...] = (".precomputed", "precomputed")
# Only fall back to enumerating every subdirectory when there are very few; otherwise a shared ``pytest-*``
# tmp root (or ``$HOME``) can accidentally match another tree. Otherwise we narrow to the manifest path + hints.
_FULL_SIBLING_SCAN_MAX = 32
# When a parent has too many subdirectories, still check these names (if present) for cousin preprocess trees.
_PRECOMPUTED_SIBLING_HINT_NAMES: tuple[str, ...] = (
    "outputs",
    "output",
    "preprocessed",
    "preprocess",
    "export",
    "exports",
    "data",
    "datasets",
    "tensor_cache",
    "cache",
    "artifacts",
)


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Resolve the directory that contains ``latents/`` and ``conditions/`` (or their ``.precomputed`` parent).

        Accepts either:
        - A **dataset / archive root** that contains a ``.precomputed/`` folder (trainer config style), or
        - A path inside the tree (e.g. ``.../ltx_manifest``) with tensors stored under ``.precomputed`` on an ancestor.

        Walks from ``data_root`` upward (limited depth). At each directory it checks, in order:

        - ``<dir>/.precomputed`` or ``<dir>/precomputed`` (must contain ``latents/``),
        - ``<dir>/latents`` + ``<dir>/conditions`` (flat layout),
        - **Child folders** of ``<dir>`` (cousin layout), e.g. ``archive/run/.precomputed`` next to
          ``archive/ltx_manifest``. If ``parent`` has more than ``_FULL_SIBLING_SCAN_MAX`` children, only the
          subdirectory that continues the path toward ``data_root`` plus a small set of common names are scanned
          (so ``$HOME`` with many folders does not skip ``~/datasets/...``).
        """
        data_root_p = Path(data_root).expanduser().resolve()

        if not data_root_p.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root_p}")

        cur = data_root_p
        for step in range(25):
            hit = PrecomputedDataset._resolve_precomputed_at_dir(cur, data_root_p, step)
            if hit is not None:
                return hit
            hit = PrecomputedDataset._resolve_precomputed_in_child_dirs(cur, data_root_p)
            if hit is not None:
                return hit
            if cur == cur.parent:
                break
            cur = cur.parent

        raise FileNotFoundError(
            f"No {_PRECOMPUTED_DIR_ALIASES[0]!r} / {_PRECOMPUTED_DIR_ALIASES[1]!r} (with latents/) or flat "
            f"latents/ + conditions/ found starting from {data_root_p} "
            f"(searched this path, up to 24 parents, and cousin folders — full scan when ≤{_FULL_SIBLING_SCAN_MAX} "
            f"subdirs, else narrowed along your manifest path + common names). "
            "Run dataset preprocessing (latents + caption embeddings), or set ``data.preprocessed_data_root`` to the "
            "folder that contains preprocess output (often a sibling of ``ltx_manifest/`` such as "
            "``outputs/.precomputed``). If you changed the Gemma / LTX checkpoint, re-run ``process_captions`` so "
            "``conditions/*.pt`` match the same encoder stack as training."
        )

    @staticmethod
    def _precomputed_tree_with_latents(path: Path) -> Path | None:
        if not path.is_dir():
            return None
        return path if (path / "latents").is_dir() else None

    @staticmethod
    def _find_named_precomputed_under(path: Path) -> Path | None:
        for name in _PRECOMPUTED_DIR_ALIASES:
            pc = path / name
            hit = PrecomputedDataset._precomputed_tree_with_latents(pc)
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _find_flat_latents_under(path: Path) -> Path | None:
        if (path / "latents").is_dir() and (path / "conditions").is_dir():
            return path
        return None

    @staticmethod
    def _resolve_precomputed_at_dir(cur: Path, data_root_p: Path, step: int) -> Path | None:
        hit = PrecomputedDataset._find_named_precomputed_under(cur)
        if hit is not None:
            if step > 0 or cur != data_root_p:
                logger.info(
                    "Resolved precomputed data root from %s to %s (found after walking up %d level(s)).",
                    data_root_p,
                    hit,
                    step,
                )
            return hit
        hit = PrecomputedDataset._find_flat_latents_under(cur)
        if hit is not None:
            if step > 0 or cur != data_root_p:
                logger.info(
                    "Resolved precomputed data root from %s to %s (flat latents/ + conditions/ on ancestor).",
                    data_root_p,
                    cur,
                )
            return hit
        return None

    @staticmethod
    def _manifest_path_parts_under_parent(parent_dir: Path, data_root_p: Path) -> tuple[str, ...] | None:
        """Parts of *data_root_p* relative to *parent_dir* if the manifest path is under that parent.

        Uses :meth:`pathlib.Path.relative_to` when possible, and falls back to a resolved-string prefix so
        symlink-heavy layouts still narrow sibling scans (e.g. ``$HOME`` → ``datasets``).
        """
        pr, dr = parent_dir.resolve(), data_root_p.resolve()
        try:
            rel = dr.relative_to(pr)
            return tuple(rel.parts)
        except ValueError:
            pfx = str(pr)
            if not pfx.endswith(os.sep):
                pfx = pfx + os.sep
            ds = str(dr)
            if not ds.startswith(pfx):
                return None
            rest = ds[len(pfx) :].lstrip(os.sep)
            if not rest:
                return ()
            return tuple(Path(rest).parts)

    @staticmethod
    def _candidate_child_dirs_for_precomputed_scan(parent_dir: Path, data_root_p: Path) -> list[Path]:
        """Subdirectories of ``parent_dir`` to inspect for ``<child>/.precomputed`` (cousin layout)."""
        if not parent_dir.is_dir():
            return []
        try:
            kids = sorted((p for p in parent_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
        except OSError:
            return []

        out: list[Path] = []
        seen: set[str] = set()

        def _add(p: Path) -> None:
            if not p.is_dir():
                return
            key = str(p.resolve())
            if key in seen:
                return
            seen.add(key)
            out.append(p)

        rel_parts = PrecomputedDataset._manifest_path_parts_under_parent(parent_dir, data_root_p)
        if rel_parts:
            _add(parent_dir / rel_parts[0])

        for name in _PRECOMPUTED_SIBLING_HINT_NAMES:
            _add(parent_dir / name)

        # Cousin layout: tensors may live under ``outputs/.precomputed`` or ``tensor_cache/.precomputed`` while the
        # manifest is only under ``ltx_manifest/``. We must not return as soon as ``relative_to`` adds the manifest
        # segment — still scan siblings for a valid subtree. When the parent has very many children, cap the scan.
        heavy = len(kids) > _FULL_SIBLING_SCAN_MAX
        cap = min(128, len(kids)) if heavy else len(kids)
        for sub in kids[:cap]:
            if PrecomputedDataset._find_named_precomputed_under(sub) or PrecomputedDataset._find_flat_latents_under(sub):
                _add(sub)

        if out:
            if heavy:
                logger.info(
                    "Narrowed precomputed sibling scan under %s (%d subdirectories) to %d candidate folder(s).",
                    parent_dir,
                    len(kids),
                    len(out),
                )
            return out

        if len(kids) <= 8:
            return kids

        logger.warning(
            "Skipping precomputed sibling scan under %s (%d subdirectories) — could not narrow along the "
            "configured data path; set ``data.preprocessed_data_root`` explicitly.",
            parent_dir,
            len(kids),
        )
        return []

    @staticmethod
    def _resolve_precomputed_in_child_dirs(
        parent_dir: Path, data_root_p: Path, *, _depth: int = 0
    ) -> Path | None:
        """Look for ``<child>/.precomputed`` or flat tensors under subdirectories of ``parent_dir``.

        When ``parent`` has many children, :meth:`_candidate_child_dirs_for_precomputed_scan` narrows the set.
        If a candidate is an intermediate directory (e.g. ``datasets`` on the way to ``.../ltx_manifest``),
        recurse one level so ``.../archive/outputs/.precomputed`` is still discoverable.
        """
        if _depth > 8:
            return None
        for sub in PrecomputedDataset._candidate_child_dirs_for_precomputed_scan(parent_dir, data_root_p):
            hit = PrecomputedDataset._find_named_precomputed_under(sub)
            if hit is not None:
                logger.info(
                    "Resolved precomputed data root from %s to %s (found under sibling folder %s).",
                    data_root_p,
                    hit,
                    sub.name,
                )
                return hit
            hit = PrecomputedDataset._find_flat_latents_under(sub)
            if hit is not None:
                logger.info(
                    "Resolved precomputed data root from %s to %s (flat tensors under sibling folder %s).",
                    data_root_p,
                    hit,
                    sub.name,
                )
                return hit
            nested = PrecomputedDataset._resolve_precomputed_in_child_dirs(sub, data_root_p, _depth=_depth + 1)
            if nested is not None:
                return nested
        return None

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Required {dir_name!r} directory does not exist: {source_path}\n"
                    f"(resolved precomputed data root: {self.data_root}). "
                    "Expected preprocess output under this root: latents/ and conditions/ "
                    f"(often inside a {PRECOMPUTED_DIR_NAME}/ or precomputed/ folder next to your dataset manifest)."
                )

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources.
        Uses a fast two-pass approach: first globs all sources in parallel to build
        full-path sets in memory, then checks expected paths via set membership.
        This avoids O(N * num_sources) stat calls on networked filesystems while
        correctly handling path remapping (e.g. latent_X.pt -> condition_X.pt).
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]

        # Pass 1: Glob all sources in parallel, build full-path sets
        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            source_path = self.source_paths[dir_name]
            paths = list(source_path.glob("**/*.pt"))
            path_set = {str(p) for p in paths}
            return paths, path_set

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        # Get primary source files (cached from glob, no second scan)
        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        # Log source sizes
        for dir_name, (paths, _) in glob_results.items():
            logger.debug(f"Source {dir_name}: {len(paths)} files")

        # Build path sets for non-primary sources
        other_path_sets = {
            dir_name: path_set for dir_name, (_, path_set) in glob_results.items() if dir_name != data_key
        }

        # Pass 2: For each primary file, check if expected paths exist in other sources' sets
        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        valid_count = 0

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check all other sources via set lookup (O(1) per source, no stat calls)
            all_exist = True
            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(f"Skipping {data_file.name}: no matching {dir_name} file at {expected}")
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(f"Fast index: {valid_count} valid samples from {len(data_files)} total ({skipped} skipped)")
        else:
            logger.debug(f"Fast index: {valid_count} valid samples from {len(data_files)} total")

        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )

        # Verify all output keys have the same number of samples
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data
