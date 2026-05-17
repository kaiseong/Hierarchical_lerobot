#!/usr/bin/env python3
"""Phase-0 SAM3 filtering test for LeRobot datasets.

This script loads a LeRobot dataset, applies camera-specific SAM3 prompt/BBOX
segmentation rules from a JSON config, and saves per-camera filtered RGB images
where only the union of selected role masks remains and the background is black.

It is intentionally a *test/inspection* tool, not the final dataset converter.
Use `--mock` on machines without SAM3/GPU to validate dataset keys, BBOX config,
and output layout. Run without `--mock` on the RTX 5090/Thor machine.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class RoleResult:
    name: str
    mask: np.ndarray
    scores: list[float]
    boxes_xyxy: list[list[float]]
    prompts: list[str]
    elapsed_ms: float
    missing: bool = False
    error: str | None = None


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def scalar_to_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def tensor_or_array_to_uint8_hwc(value: Any) -> np.ndarray:
    """Convert LeRobot image tensor/array to uint8 HWC RGB."""
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim == 4:
        # Delta timestamp windows are not expected for this tool; use current frame if present.
        arr = arr[-1]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor/array, got shape {arr.shape}")

    # LeRobot tensors are normally CHW. Images loaded outside torch may be HWC.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        finite = np.isfinite(arr)
        if finite.any() and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def normalized_xyxy_to_mask(box: list[float], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    x0 = int(round(np.clip(x0, 0, 1) * width))
    x1 = int(round(np.clip(x1, 0, 1) * width))
    y0 = int(round(np.clip(y0, 0, 1) * height))
    y1 = int(round(np.clip(y1, 0, 1) * height))
    mask = np.zeros((height, width), dtype=bool)
    mask[min(y0, y1) : max(y0, y1), min(x0, x1) : max(x0, x1)] = True
    return mask


def box_cfg_to_normalized_xyxy(box_cfg: dict[str, Any], height: int, width: int) -> list[float]:
    fmt = box_cfg.get("format", "normalized_xyxy")
    val = [float(v) for v in box_cfg["value"]]
    if fmt == "normalized_xyxy":
        return val
    if fmt == "normalized_cxcywh":
        cx, cy, bw, bh = val
        return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
    if fmt == "pixel_xyxy":
        x0, y0, x1, y1 = val
        return [x0 / width, y0 / height, x1 / width, y1 / height]
    if fmt == "pixel_xywh":
        x, y, bw, bh = val
        return [x / width, y / height, (x + bw) / width, (y + bh) / height]
    raise ValueError(f"Unsupported box format: {fmt}")


def regions_to_mask(regions: list[dict[str, Any]] | None, height: int, width: int) -> np.ndarray | None:
    if not regions:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        mask |= normalized_xyxy_to_mask(box_cfg_to_normalized_xyxy(region, height, width), height, width)
    return mask


def box_to_normalized_cxcywh(box_cfg: dict[str, Any], height: int, width: int) -> list[float]:
    fmt = box_cfg.get("format", "normalized_xyxy")
    val = [float(v) for v in box_cfg["value"]]
    if fmt == "normalized_xyxy":
        x0, y0, x1, y1 = val
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        return [cx, cy, abs(x1 - x0), abs(y1 - y0)]
    if fmt == "normalized_cxcywh":
        return val
    if fmt == "pixel_xyxy":
        x0, y0, x1, y1 = val
        cx = ((x0 + x1) / 2.0) / width
        cy = ((y0 + y1) / 2.0) / height
        return [cx, cy, abs(x1 - x0) / width, abs(y1 - y0) / height]
    if fmt == "pixel_xywh":
        x, y, w, h = val
        return [(x + w / 2.0) / width, (y + h / 2.0) / height, w / width, h / height]
    raise ValueError(f"Unsupported box format: {fmt}")


def output_box_to_xyxy_list(box: Any) -> list[float]:
    if hasattr(box, "detach"):
        box = box.detach().cpu().tolist()
    elif hasattr(box, "tolist"):
        box = box.tolist()
    return [float(x) for x in box]


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or not mask.any():
        return mask
    # Prefer cv2 when available; fall back to PIL MaxFilter.
    try:
        import cv2  # type: ignore

        kernel_size = max(1, radius_px * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        img = Image.fromarray(mask.astype(np.uint8) * 255)
        # MaxFilter size must be odd.
        size = max(3, radius_px * 2 + 1)
        if size % 2 == 0:
            size += 1
        return np.asarray(img.filter(ImageFilter.MaxFilter(size))) > 0  # type: ignore[name-defined]


def apply_keep_mask(image: np.ndarray, mask: np.ndarray, background_value: int = 0) -> np.ndarray:
    out = np.full_like(image, background_value, dtype=np.uint8)
    out[mask] = image[mask]
    return out


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    color = np.zeros_like(overlay)
    color[..., 1] = 255  # green
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color[mask]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_config_boxes(image: np.ndarray, roles: list[dict[str, Any]]) -> np.ndarray:
    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    h, w = image.shape[:2]
    colors = ["red", "yellow", "cyan", "magenta", "orange", "lime"]
    ci = 0
    for role in roles:
        all_regions = []
        for box in role.get("boxes", []) or []:
            b = dict(box)
            b["_kind"] = "prompt_box"
            all_regions.append(b)
        for box in role.get("search_regions", []) or []:
            b = dict(box)
            b["_kind"] = "search_region"
            all_regions.append(b)
        for box in all_regions:
            fmt = box.get("format", "normalized_xyxy")
            val = [float(v) for v in box["value"]]
            if fmt == "normalized_xyxy":
                x0, y0, x1, y1 = val[0] * w, val[1] * h, val[2] * w, val[3] * h
            elif fmt == "normalized_cxcywh":
                cx, cy, bw, bh = val
                x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            elif fmt == "pixel_xyxy":
                x0, y0, x1, y1 = val
            elif fmt == "pixel_xywh":
                x0, y0, bw, bh = val
                x1, y1 = x0 + bw, y0 + bh
            else:
                continue
            color = colors[ci % len(colors)]
            ci += 1
            width_px = 3 if box.get("_kind") == "prompt_box" else 2
            label = role.get("name", "role")
            if box.get("_kind") == "search_region":
                label = f"{label}:search"
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width_px)
            draw.text((x0 + 4, y0 + 4), label, fill=color)
    return np.asarray(pil)


class MockSegmenter:
    def __init__(self, mock_text_mask: str = "center") -> None:
        self.mock_text_mask = mock_text_mask

    def segment_role(self, image: np.ndarray, role: dict[str, Any]) -> RoleResult:
        start = time.perf_counter()
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        boxes = role.get("boxes", []) or []
        search_mask = regions_to_mask(role.get("search_regions"), h, w)
        for box in boxes:
            if not bool(box.get("label", True)):
                continue
            fmt = box.get("format", "normalized_xyxy")
            if fmt == "normalized_xyxy":
                mask |= normalized_xyxy_to_mask([float(v) for v in box["value"]], h, w)
            else:
                cx, cy, bw, bh = box_to_normalized_cxcywh(box, h, w)
                mask |= normalized_xyxy_to_mask([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], h, w)
        if search_mask is not None and not boxes:
            mask |= search_mask
        if not boxes and search_mask is None and self.mock_text_mask == "center":
            # Fake a text-only object mask so output plumbing can be inspected locally.
            mask |= normalized_xyxy_to_mask([0.35, 0.35, 0.65, 0.65], h, w)
        elapsed = (time.perf_counter() - start) * 1000
        return RoleResult(
            name=role["name"],
            mask=mask,
            scores=[1.0] if mask.any() else [],
            boxes_xyxy=[],
            prompts=role.get("prompts", []),
            elapsed_ms=elapsed,
            missing=not mask.any(),
        )


class SAM3ImageSegmenter:
    def __init__(self, confidence_threshold: float, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.torch = torch
        self.device = device
        self.dtype = dtype
        if device == "cuda":
            if dtype == "bfloat16":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            elif dtype == "float16":
                torch.autocast("cuda", dtype=torch.float16).__enter__()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def _extract_masks(
        self,
        state: dict[str, Any],
        image_shape: tuple[int, int],
        max_instances: int,
        search_mask: np.ndarray | None = None,
        restrict_to_search_region: bool = False,
    ) -> tuple[np.ndarray, list[float], list[list[float]]]:
        h, w = image_shape
        union = np.zeros((h, w), dtype=bool)
        scores_out: list[float] = []
        boxes_out: list[list[float]] = []
        masks = state.get("masks")
        scores = state.get("scores")
        boxes = state.get("boxes")
        if masks is None or scores is None:
            return union, scores_out, boxes_out
        if hasattr(scores, "detach"):
            scores_list = scores.detach().cpu().tolist()
        else:
            scores_list = list(scores)
        order = sorted(range(len(scores_list)), key=lambda i: float(scores_list[i]), reverse=True)
        kept = 0
        for i in order:
            score = float(scores_list[i])
            if score < self.confidence_threshold:
                continue
            mask_i = masks[i]
            if hasattr(mask_i, "detach"):
                mask_np = mask_i.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask_i)
            mask_np = np.squeeze(mask_np).astype(bool)
            if mask_np.shape != (h, w):
                mask_np = np.asarray(Image.fromarray(mask_np.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
            if search_mask is not None:
                if not np.logical_and(mask_np, search_mask).any():
                    continue
                if restrict_to_search_region:
                    mask_np = np.logical_and(mask_np, search_mask)
            union |= mask_np
            scores_out.append(score)
            if boxes is not None:
                boxes_out.append(output_box_to_xyxy_list(boxes[i]))
            kept += 1
            if kept >= max_instances:
                break
        return union, scores_out, boxes_out

    def segment_role(self, image: np.ndarray, role: dict[str, Any]) -> RoleResult:
        start = time.perf_counter()
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        prompts = role.get("prompts", []) or [None]
        boxes = role.get("boxes", []) or []
        max_instances = int(role.get("max_instances", role.get("max_instances_per_role", 3)))
        search_mask = regions_to_mask(role.get("search_regions"), h, w)
        restrict_to_search_region = bool(role.get("restrict_to_search_region", False))
        role_mask = np.zeros((h, w), dtype=bool)
        role_scores: list[float] = []
        role_boxes: list[list[float]] = []
        error = None

        try:
            for prompt in prompts:
                state = self.processor.set_image(pil)
                if prompt:
                    state = self.processor.set_text_prompt(state=state, prompt=str(prompt))
                for box_cfg in boxes:
                    box = box_to_normalized_cxcywh(box_cfg, h, w)
                    label = bool(box_cfg.get("label", True))
                    try:
                        state = self.processor.add_geometric_prompt(
                            state=state,
                            box=box,
                            label=label,
                            text_prompt=str(prompt) if prompt else None,
                        )
                    except TypeError:
                        state = self.processor.add_geometric_prompt(state=state, box=box, label=label)
                mask, scores, out_boxes = self._extract_masks(
                    state,
                    (h, w),
                    max_instances=max_instances,
                    search_mask=search_mask,
                    restrict_to_search_region=restrict_to_search_region,
                )
                role_mask |= mask
                role_scores.extend(scores)
                role_boxes.extend(out_boxes)
        except Exception as exc:  # keep batch running and log per-role failures
            error = repr(exc)

        elapsed = (time.perf_counter() - start) * 1000
        return RoleResult(
            name=role["name"],
            mask=role_mask,
            scores=role_scores,
            boxes_xyxy=role_boxes,
            prompts=[p for p in role.get("prompts", [])],
            elapsed_ms=elapsed,
            missing=not role_mask.any(),
            error=error,
        )


def build_segmenter(config: dict[str, Any], mock: bool, mock_text_mask: str):
    if mock:
        return MockSegmenter(mock_text_mask=mock_text_mask)
    sam_cfg = config.get("sam3", {})
    return SAM3ImageSegmenter(
        confidence_threshold=float(sam_cfg.get("confidence_threshold", 0.5)),
        device=str(sam_cfg.get("device", "cuda")),
        dtype=str(sam_cfg.get("dtype", "bfloat16")),
    )


def process_camera(
    image: np.ndarray,
    camera_alias: str,
    camera_cfg: dict[str, Any],
    segmenter: Any,
    sam_cfg: dict[str, Any],
    out_dir: Path,
    frame_stem: str,
    save_role_masks: bool,
) -> dict[str, Any]:
    roles = camera_cfg.get("roles", [])
    keep = np.zeros(image.shape[:2], dtype=bool)
    role_logs: list[dict[str, Any]] = []
    role_dir = out_dir / "role_masks"
    if save_role_masks:
        role_dir.mkdir(parents=True, exist_ok=True)

    for role in roles:
        result = segmenter.segment_role(image, role)
        role_mask = result.mask
        keep |= role_mask
        if save_role_masks:
            Image.fromarray(role_mask.astype(np.uint8) * 255).save(role_dir / f"{frame_stem}_{camera_alias}_{result.name}.png")
        role_logs.append(
            {
                "name": result.name,
                "prompts": result.prompts,
                "scores": result.scores,
                "boxes_xyxy": result.boxes_xyxy,
                "search_regions": role.get("search_regions", []),
                "elapsed_ms": round(result.elapsed_ms, 3),
                "missing": result.missing,
                "optional": bool(role.get("optional", False)),
                "error": result.error,
            }
        )

    dilation_px = int(sam_cfg.get("mask_dilation_px", 0))
    keep = dilate_mask(keep, dilation_px)
    filtered = apply_keep_mask(image, keep, int(sam_cfg.get("background_value", 0)))
    overlay = make_overlay(image, keep)
    boxes_preview = draw_config_boxes(image, roles)

    Image.fromarray(filtered).save(out_dir / f"{frame_stem}_{camera_alias}_filtered.png")
    Image.fromarray(overlay).save(out_dir / f"{frame_stem}_{camera_alias}_overlay.png")
    Image.fromarray(keep.astype(np.uint8) * 255).save(out_dir / f"{frame_stem}_{camera_alias}_keep_mask.png")
    Image.fromarray(boxes_preview).save(out_dir / f"{frame_stem}_{camera_alias}_config_boxes.png")

    return {
        "camera": camera_alias,
        "keep_coverage": float(keep.mean()),
        "roles": role_logs,
        "outputs": {
            "filtered": f"{frame_stem}_{camera_alias}_filtered.png",
            "overlay": f"{frame_stem}_{camera_alias}_overlay.png",
            "keep_mask": f"{frame_stem}_{camera_alias}_keep_mask.png",
            "config_boxes": f"{frame_stem}_{camera_alias}_config_boxes.png",
        },
    }


def run(args: argparse.Namespace) -> None:
    from lerobot.datasets import LeRobotDataset

    config = load_json(args.config)
    ds_cfg = config.get("dataset", {})
    repo_id = args.repo_id or ds_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("Dataset repo_id must be provided by config.dataset.repo_id or --repo-id")
    root = args.root if args.root is not None else ds_cfg.get("root")
    episodes = parse_episodes(args.episodes) if args.episodes is not None else ds_cfg.get("episodes")
    if episodes is not None:
        episodes = [int(e) for e in episodes]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        return_uint8=True,
        download_videos=not args.no_download_videos,
    )
    camera_keys = ds_cfg.get("camera_keys", {})
    segmenter = build_segmenter(config, mock=args.mock, mock_text_mask=args.mock_text_mask)

    manifest_path = out_dir / "manifest.jsonl"
    summary = {
        "repo_id": repo_id,
        "root": str(root) if root else None,
        "episodes": episodes,
        "config": str(args.config),
        "mock": args.mock,
        "frames_processed": 0,
        "camera_keys": camera_keys,
    }

    per_episode_counts: dict[int, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for rel_idx in range(len(dataset)):
            item = dataset[rel_idx]
            ep_idx = scalar_to_int(item.get("episode_index", 0))
            if episodes is not None and ep_idx not in episodes:
                continue
            count = per_episode_counts.get(ep_idx, 0)
            if args.max_frames_per_episode is not None and count >= args.max_frames_per_episode:
                continue
            if count % args.frame_stride != 0:
                per_episode_counts[ep_idx] = count + 1
                continue
            per_episode_counts[ep_idx] = count + 1

            frame_idx = scalar_to_int(item.get("frame_index", item.get("index", rel_idx)))
            frame_stem = f"ep{ep_idx:06d}_frame{frame_idx:06d}"
            frame_log = {
                "relative_index": rel_idx,
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "task": str(item.get("task", "")),
                "cameras": [],
            }

            for camera_alias, camera_cfg in config.get("cameras", {}).items():
                dataset_key = camera_keys.get(camera_alias, camera_cfg.get("dataset_key"))
                if not dataset_key:
                    frame_log["cameras"].append({"camera": camera_alias, "error": "missing dataset_key"})
                    continue
                if dataset_key not in item:
                    frame_log["cameras"].append(
                        {"camera": camera_alias, "dataset_key": dataset_key, "error": "key not present in dataset item"}
                    )
                    continue
                image = tensor_or_array_to_uint8_hwc(item[dataset_key])
                cam_log = process_camera(
                    image=image,
                    camera_alias=camera_alias,
                    camera_cfg=camera_cfg,
                    segmenter=segmenter,
                    sam_cfg=config.get("sam3", {}),
                    out_dir=out_dir,
                    frame_stem=frame_stem,
                    save_role_masks=args.save_role_masks,
                )
                cam_log["dataset_key"] = dataset_key
                frame_log["cameras"].append(cam_log)

            manifest.write(json.dumps(frame_log, ensure_ascii=False) + "\n")
            summary["frames_processed"] += 1
            if args.max_total_frames is not None and summary["frames_processed"] >= args.max_total_frames:
                break

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": summary, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to SAM3 filtering JSON config")
    parser.add_argument("--repo-id", default=None, help="Override config.dataset.repo_id")
    parser.add_argument("--root", default=None, help="Override config.dataset.root for local LeRobot dataset")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices, e.g. 0,1,2")
    parser.add_argument("--output-dir", default="examples/sam3_filtering/outputs/run", help="Directory for filtered/overlay outputs")
    parser.add_argument("--max-frames-per-episode", type=int, default=10, help="Limit frames processed per episode")
    parser.add_argument("--max-total-frames", type=int, default=None, help="Global frame limit")
    parser.add_argument("--frame-stride", type=int, default=15, help="Process every Nth frame within each selected episode")
    parser.add_argument("--no-download-videos", action="store_true", help="Do not download missing dataset videos")
    parser.add_argument("--mock", action="store_true", help="Use BBOX/dummy masks instead of importing/running SAM3")
    parser.add_argument(
        "--mock-text-mask",
        choices=["blank", "center"],
        default="center",
        help="Mock behavior for text-only roles without boxes",
    )
    parser.add_argument("--save-role-masks", action="store_true", help="Save individual role masks in addition to union masks")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
