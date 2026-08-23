# Third-party upstream trees

## `nv-sana` — [NVlabs/Sana](https://github.com/NVlabs/Sana)

Apache-2.0 NVIDIA research codebase for efficient image/video diffusion (SANA, SANA-Video, LongSANA, SANA-WM, schedulers, training scripts). Kino keeps this as a **git submodule** so we can diff against upstream, borrow sampling/long-horizon patterns, and track releases without forking the full tree into `packages/`.

### Refresh

```bash
git submodule update --init --recursive
# or pull the latest upstream commit on the tracked branch:
cd third_party/nv-sana && git fetch origin && git checkout main && git pull
cd ../.. && git add third_party/nv-sana && git commit -m "Bump nv-sana submodule"
```

Attribution: see `nv-sana/LICENSE` and the papers linked from the [SANA docs](https://nvlabs.github.io/Sana/docs/).
