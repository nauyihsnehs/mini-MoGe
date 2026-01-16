# MoGe Minimal Depth Inference (v2)

This repository is trimmed down to the minimal pieces required to run **depth-only** inference with MoGe v2.
It provides:

- A **core model wrapper** (`moge/minimal_depth_model.py`) for depth + FoV inference.
- A **batch inference class** (`moge/depth_batch_inferencer.py`) for tensor batches.
- A **CLI script** (`moge/scripts/infer.py`) for folder or single-image inference.

## Quick start

### CLI

```bash
python moge/scripts/infer.py \
  -i /path/to/images \
  -o /path/to/output \
  --maps
```

Optional arguments:
- `--pretrained` (default: `Ruicheng/moge-2-vitl`)
- `--device` (default: `cuda`)
- `--fp16`
- `--resize`
- `--resolution_level` / `--num_tokens`
- `--batch_size`

### In-code usage

```python
import torch
from moge.depth_batch_inferencer import DepthBatchInferencer

inferencer = DepthBatchInferencer.from_pretrained("Ruicheng/moge-2-vitl", device="cuda")
images = torch.randn(4, 3, 512, 512, device="cuda")
output = inferencer.infer_batch(images)

# output keys: depth, fov_x, fov_y
```

## Outputs

For each input image, the CLI writes:

- `depth.exr` (float32 depth map)
- `fov.json` (estimated FoV values)
