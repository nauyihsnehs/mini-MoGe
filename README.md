# MoGe Minimal Depth Inference (v2)

This repository is trimmed down to the minimal pieces required to run **depth-only** inference with MoGe v2.
It provides a single inference module (`moge/inference.py`) with both depth and batch helpers.

## Quick start

### In-code usage

```python
import torch
from moge.inference import DepthBatchInferencer

inferencer = DepthBatchInferencer.from_pretrained("Ruicheng/moge-2-vitl", device="cuda")
images = torch.randn(4, 3, 512, 512, device="cuda")
output = inferencer.infer_batch(images)

# output keys: depth, fov_x, fov_y
```

## Outputs

The inference output includes:

- `depth` (float32 depth map tensor)
- `fov_x` / `fov_y` (estimated FoV values)
