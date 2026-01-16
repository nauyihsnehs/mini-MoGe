from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch

from moge.minimal_depth_model import MoGeDepthModel


class DepthBatchInferencer:
    def __init__(self, model: MoGeDepthModel) -> None:
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        use_fp16: bool = False,
        **hf_kwargs,
    ) -> "DepthBatchInferencer":
        model = MoGeDepthModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            use_fp16=use_fp16,
            **hf_kwargs,
        )
        return cls(model)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def infer_batch(
        self,
        images: torch.Tensor,
        fov_x: Optional[float] = None,
        resolution_level: int = 9,
        num_tokens: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model.infer_depth(
            images,
            fov_x=fov_x,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )
