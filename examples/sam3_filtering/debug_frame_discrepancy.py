#!/usr/bin/env python3
"""Compare standalone SAM3 image results with LeRobot episode filtering inputs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def maybe_add_sam3_path(path: str | None) -> None:
    candidates = [Path(path).expanduser()] if path else []
    candidates.extend([Path("/home/kgs/sam3"), Path("/home/rby1/sam3")])
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Standalone image path to compare, e.g. front_107.jpg.")
    parser.add_argument("--prompt", default="robot", help="SAM3 text prompt to test.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Standalone processor threshold.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--repo-id", default="rainbowrobotics/simtos_0412")
    parser.add_argument("--dataset-root", default=None, help="Optional local LeRobot dataset root/snapshot.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=107)
    parser.add_argument("--camera-key", default="observation.images.front")
    parser.add_argument("--config", required=True, help="Filtering JSON config used for the episode run.")
    parser.add_argument("--camera-alias", default="front")
    parser.add_argument("--role-name", default="robot_arm_gripper")
    parser.add_argument("--sam3-root", default=None, help="Optional SAM3 repo path if sam3 is not installed.")
    parser.add_argument("--out-dir", default="examples/sam3_filtering/outputs/debug_frame_discrepancy")
    return parser.parse_args()


def autocast_context(precision: str):
    if precision == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_scores(value: Any) -> list[float]:
    return [float(x) for x in to_numpy(value).reshape(-1).tolist()]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def image_diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_mismatch": True}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "shape": list(a.shape),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "nonzero_pct": float((diff > 0).mean() * 100.0),
    }


def run_direct_processor(
    model: Any,
    processor_cls: Any,
    image: Image.Image,
    prompt: str,
    threshold: float,
    precision: str,
) -> dict[str, Any]:
    processor = processor_cls(model, resolution=1008, device="cuda", confidence_threshold=threshold)
    start = time.perf_counter()
    with torch.inference_mode(), autocast_context(precision):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        out = processor.set_text_prompt(prompt=prompt, state=state)
    torch.cuda.synchronize()
    scores = normalize_scores(out.get("scores"))
    boxes = to_numpy(out.get("boxes"))
    masks = to_numpy(out.get("masks"))
    return {
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "num_scores": len(scores),
        "scores": [round(score, 6) for score in scores],
        "boxes_shape": list(boxes.shape),
        "masks_shape": list(masks.shape),
    }


def find_role(config: dict[str, Any], camera_alias: str, role_name: str) -> dict[str, Any]:
    for role in config.get("cameras", {}).get(camera_alias, {}).get("roles", []):
        if role.get("name") == role_name:
            return role
    raise ValueError(f"Role {camera_alias}.{role_name} not found")


def run_filter_segmenter(
    filter_module: Any,
    image_array: np.ndarray,
    config: dict[str, Any],
    camera_alias: str,
    role_name: str,
) -> dict[str, Any]:
    role = find_role(config, camera_alias, role_name)
    default_threshold, processor_threshold = filter_module.config_confidence_thresholds(config)
    sam_cfg = config.get("sam3", {})
    segmenter = filter_module.SAM3ImageSegmenter(
        confidence_threshold=default_threshold,
        processor_confidence_threshold=processor_threshold,
        device=str(sam_cfg.get("device", "cuda")),
        dtype=str(sam_cfg.get("dtype", "bfloat16")),
    )
    result = segmenter.segment_role(image_array, role)
    payload = {
        "role": role,
        "default_threshold": default_threshold,
        "processor_threshold": processor_threshold,
        "result_threshold": result.confidence_threshold,
        "elapsed_ms": round(result.elapsed_ms, 3),
        "scores": [round(float(score), 6) for score in result.scores],
        "num_scores": len(result.scores),
        "boxes_xyxy": result.boxes_xyxy,
        "missing": result.missing,
        "error": result.error,
        "mask_coverage": float(result.mask.mean()),
    }
    del segmenter
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> int:
    args = parse_args()
    maybe_add_sam3_path(args.sam3_root)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from lerobot.datasets import LeRobotDataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on the GPU machine.")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    filter_module = load_module(Path(__file__).with_name("sam3_filter_dataset.py"), "sam3_filter_dataset_debug")
    config = filter_module.load_json(args.config)

    standalone_path = Path(args.image).expanduser().resolve()
    standalone_img = Image.open(standalone_path).convert("RGB")

    ds_kwargs = {
        "repo_id": args.repo_id,
        "episodes": [args.episode],
        "return_uint8": True,
        "download_videos": False,
    }
    if args.dataset_root:
        ds_kwargs["root"] = Path(args.dataset_root).expanduser().resolve()
    dataset = LeRobotDataset(**ds_kwargs)
    item = dataset[args.frame_index]
    dataset_array = filter_module.tensor_or_array_to_uint8_hwc(item[args.camera_key])

    dataset_png = out_dir / f"dataset_frame{args.frame_index:06d}.png"
    dataset_jpg = out_dir / f"dataset_frame{args.frame_index:06d}.jpg"
    Image.fromarray(dataset_array).save(dataset_png)
    Image.fromarray(dataset_array).save(dataset_jpg, quality=95)

    variants = {
        "standalone_file": standalone_img,
        "dataset_direct_array": Image.fromarray(dataset_array),
        "dataset_saved_png": Image.open(dataset_png).convert("RGB"),
        "dataset_saved_jpg": Image.open(dataset_jpg).convert("RGB"),
    }

    metadata = {
        "standalone_image": str(standalone_path),
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "episode": args.episode,
        "requested_frame_index": args.frame_index,
        "item_episode_index": str(item.get("episode_index")),
        "item_frame_index": str(item.get("frame_index")),
        "camera_key": args.camera_key,
        "prompt": args.prompt,
        "standalone_threshold": args.threshold,
        "precision": args.precision,
        "config": str(args.config),
        "camera_alias": args.camera_alias,
        "role_name": args.role_name,
    }
    print("metadata:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    standalone_array = np.asarray(standalone_img)
    comparisons = {}
    print("image comparisons against standalone_file:")
    for name, image in variants.items():
        if name == "standalone_file":
            continue
        comparisons[name] = image_diff_stats(standalone_array, np.asarray(image))
        print(name, json.dumps(comparisons[name], ensure_ascii=False))

    print("building direct SAM3 model...")
    model = build_sam3_image_model(device="cuda")
    direct_results = {}
    for name, image in variants.items():
        direct_results[name] = run_direct_processor(
            model,
            Sam3Processor,
            image,
            args.prompt,
            args.threshold,
            args.precision,
        )
        print("direct", name, json.dumps(direct_results[name], ensure_ascii=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    filter_results = {
        "standalone_file": run_filter_segmenter(
            filter_module,
            np.asarray(standalone_img),
            config,
            args.camera_alias,
            args.role_name,
        ),
        "dataset_direct_array": run_filter_segmenter(
            filter_module,
            dataset_array,
            config,
            args.camera_alias,
            args.role_name,
        ),
    }
    for name, result in filter_results.items():
        print("filter", name, json.dumps(result, ensure_ascii=False))

    report = {
        "metadata": metadata,
        "image_comparisons": comparisons,
        "direct": direct_results,
        "filter": filter_results,
    }
    report_path = out_dir / "sam3_discrepancy_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
