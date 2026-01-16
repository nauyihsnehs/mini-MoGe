import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
import sys

if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import itertools
import json
import warnings

import click


@click.command(help="Minimal depth inference script (v2 only).")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    help='Input image or folder path. "jpg" and "png" are supported.',
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="./output",
    type=click.Path(),
    help="Output folder path",
)
@click.option(
    "--pretrained",
    "pretrained_model_name_or_path",
    default="Ruicheng/moge-2-vitl",
    type=str,
    show_default=True,
    help="Pretrained MoGe v2 model name or local path.",
)
@click.option(
    "--fov_x",
    "fov_x_",
    type=float,
    default=None,
    help="If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.",
)
@click.option("--device", "device_name", type=str, default="cuda", help='Device name (e.g. "cuda", "cuda:0", "cpu").')
@click.option("--fp16", "use_fp16", is_flag=True, help="Use fp16 precision for faster inference.")
@click.option(
    "--resize",
    "resize_to",
    type=int,
    default=None,
    help="Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).",
)
@click.option(
    "--resolution_level",
    type=int,
    default=9,
    help="An integer [0-9] for the resolution level for inference. Defaults to 9.",
)
@click.option(
    "--num_tokens",
    type=int,
    default=None,
    help="Number of tokens used for inference. Overrides resolution_level if provided.",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for folder inference. Set to 1 when images have different sizes.",
)
@click.option("--maps/--no-maps", "save_maps", default=True, help="Save depth map and fov JSON outputs.")
def main(
    input_path: str,
    output_path: str,
    pretrained_model_name_or_path: str,
    fov_x_: float,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    batch_size: int,
    save_maps: bool,
) -> None:
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from moge.depth_batch_inferencer import DepthBatchInferencer

    device = torch.device(device_name)

    include_suffices = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    input_path_obj = Path(input_path)
    if input_path_obj.is_dir():
        image_paths = sorted(
            itertools.chain(*(input_path_obj.rglob(f"*.{suffix}") for suffix in include_suffices))
        )
    else:
        image_paths = [input_path_obj]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {input_path}")

    if resize_to is None and batch_size > 1:
        warnings.warn("Batch size is forced to 1 when resize is disabled to avoid shape mismatches.")
        batch_size = 1

    inferencer = DepthBatchInferencer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device,
        use_fp16=use_fp16,
    )

    if not save_maps:
        warnings.warn("No output selected. Use --maps to save depth and fov outputs.")

    output_root = Path(output_path)

    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        batch_images = []
        for image_path in batch_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"File {image_path} does not exist.")
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            if resize_to is not None:
                height, width = (
                    min(resize_to, int(resize_to * height / width)),
                    min(resize_to, int(resize_to * width / height)),
                )
                image = cv2.resize(image, (width, height), cv2.INTER_AREA)
            image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
            batch_images.append(image_tensor)

        batch_tensor = torch.stack(batch_images, dim=0)
        output = inferencer.infer_batch(
            batch_tensor,
            fov_x=fov_x_,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )

        depth_batch = output["depth"].detach().cpu().numpy().astype(np.float32)
        fov_x_batch = output["fov_x"].detach().cpu().numpy()
        fov_y_batch = output["fov_y"].detach().cpu().numpy()

        if not save_maps:
            continue

        for idx, image_path in enumerate(batch_paths):
            if input_path_obj.is_dir():
                relative_parent = image_path.relative_to(input_path_obj).parent
            else:
                relative_parent = Path()
            save_path = output_root / relative_parent / image_path.stem
            save_path.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(
                str(save_path / "depth.exr"),
                depth_batch[idx],
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
            )
            with open(save_path / "fov.json", "w") as f:
                json.dump(
                    {
                        "fov_x": round(float(fov_x_batch[idx]), 2),
                        "fov_y": round(float(fov_y_batch[idx]), 2),
                    },
                    f,
                )


if __name__ == "__main__":
    main()
