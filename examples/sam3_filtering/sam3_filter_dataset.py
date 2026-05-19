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
from collections import defaultdict
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    confidence_threshold: float
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


def parse_optional_frame_limit(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"all", "none", "unlimited", "-1", "0"}:
        return None
    parsed = int(text)
    if parsed < 0:
        return None
    return parsed


def parse_csv(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def role_confidence_threshold(role: dict[str, Any], default_threshold: float) -> float:
    if "confidence_threshold" in role:
        return float(role["confidence_threshold"])
    if "threshold" in role:
        return float(role["threshold"])
    return default_threshold


def config_confidence_thresholds(config: dict[str, Any]) -> tuple[float, float]:
    sam_cfg = config.get("sam3", {})
    default_threshold = float(sam_cfg.get("confidence_threshold", 0.5))
    thresholds = [default_threshold]
    for camera_cfg in config.get("cameras", {}).values():
        for role in camera_cfg.get("roles", []) or []:
            thresholds.append(role_confidence_threshold(role, default_threshold))
    return default_threshold, min(thresholds)


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


def summarize_ms(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


class TimingAccumulator:
    def __init__(self) -> None:
        self.frame_wall_ms: list[float] = []
        self.camera_wall_ms: list[float] = []
        self.camera_wall_ms_by_camera: dict[str, list[float]] = defaultdict(list)
        self.role_ms_by_camera_role: dict[str, list[float]] = defaultdict(list)

    def add_frame(self, elapsed_ms: float) -> None:
        self.frame_wall_ms.append(elapsed_ms)

    def add_camera_log(self, cam_log: dict[str, Any]) -> None:
        camera = str(cam_log.get("camera", "unknown"))
        elapsed = cam_log.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_f = float(elapsed)
            self.camera_wall_ms.append(elapsed_f)
            self.camera_wall_ms_by_camera[camera].append(elapsed_f)
        for role in cam_log.get("roles", []) or []:
            role_elapsed = role.get("elapsed_ms")
            role_name = role.get("name", "unknown")
            if isinstance(role_elapsed, (int, float)):
                self.role_ms_by_camera_role[f"{camera}.{role_name}"].append(float(role_elapsed))

    def to_summary(self, camera_workers: int, role_workers: int) -> dict[str, Any]:
        return {
            "camera_workers": camera_workers,
            "role_workers": role_workers,
            "max_concurrent_sam3_models_estimate": camera_workers if role_workers == 1 else camera_workers * role_workers,
            "frame_wall_ms": summarize_ms(self.frame_wall_ms),
            "camera_wall_ms": summarize_ms(self.camera_wall_ms),
            "camera_wall_ms_by_camera": {
                camera: summarize_ms(values) for camera, values in sorted(self.camera_wall_ms_by_camera.items())
            },
            "role_ms_by_camera_role": {
                role_key: summarize_ms(values) for role_key, values in sorted(self.role_ms_by_camera_role.items())
            },
        }


class MockSegmenter:
    def __init__(self, mock_text_mask: str = "center", confidence_threshold: float = 0.5) -> None:
        self.mock_text_mask = mock_text_mask
        self.confidence_threshold = confidence_threshold

    def segment_role(self, image: np.ndarray, role: dict[str, Any]) -> RoleResult:
        start = time.perf_counter()
        confidence_threshold = role_confidence_threshold(role, self.confidence_threshold)
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
            confidence_threshold=confidence_threshold,
            elapsed_ms=elapsed,
            missing=not mask.any(),
        )


class SAM3ImageSegmenter:
    def __init__(
        self,
        confidence_threshold: float,
        processor_confidence_threshold: float | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
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
        self.processor_confidence_threshold = (
            confidence_threshold if processor_confidence_threshold is None else processor_confidence_threshold
        )
        self.processor = Sam3Processor(self.model, confidence_threshold=self.processor_confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def _extract_masks(
        self,
        state: dict[str, Any],
        image_shape: tuple[int, int],
        max_instances: int,
        confidence_threshold: float,
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
            if score < confidence_threshold:
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
        confidence_threshold = role_confidence_threshold(role, self.confidence_threshold)
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
                    confidence_threshold=confidence_threshold,
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
            confidence_threshold=confidence_threshold,
            elapsed_ms=elapsed,
            missing=not role_mask.any(),
            error=error,
        )


def build_segmenter(config: dict[str, Any], mock: bool, mock_text_mask: str):
    default_threshold, processor_threshold = config_confidence_thresholds(config)
    if mock:
        return MockSegmenter(mock_text_mask=mock_text_mask, confidence_threshold=default_threshold)
    sam_cfg = config.get("sam3", {})
    return SAM3ImageSegmenter(
        confidence_threshold=default_threshold,
        processor_confidence_threshold=processor_threshold,
        device=str(sam_cfg.get("device", "cuda")),
        dtype=str(sam_cfg.get("dtype", "bfloat16")),
    )


class ThreadLocalSegmenterPool:
    """Keep one segmenter per worker thread so SAM3 processor state is not shared."""

    def __init__(self, config: dict[str, Any], mock: bool, mock_text_mask: str) -> None:
        self.config = config
        self.mock = mock
        self.mock_text_mask = mock_text_mask
        self.local = threading.local()

    def get(self) -> Any:
        segmenter = getattr(self.local, "segmenter", None)
        if segmenter is None:
            segmenter = build_segmenter(self.config, mock=self.mock, mock_text_mask=self.mock_text_mask)
            self.local.segmenter = segmenter
        return segmenter


def segment_role_from_pool(segmenter_pool: ThreadLocalSegmenterPool, image: np.ndarray, role: dict[str, Any]) -> RoleResult:
    return segmenter_pool.get().segment_role(image, role)


def process_camera(
    image: np.ndarray,
    camera_alias: str,
    camera_cfg: dict[str, Any],
    segmenter: Any,
    sam_cfg: dict[str, Any],
    out_dir: Path,
    images_dir: Path,
    frame_stem: str,
    image_kinds: list[str],
    save_role_masks: bool,
    role_workers: int = 1,
    role_executor: ThreadPoolExecutor | None = None,
    segmenter_pool: ThreadLocalSegmenterPool | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    roles = camera_cfg.get("roles", [])
    keep = np.zeros(image.shape[:2], dtype=bool)
    role_logs: list[dict[str, Any]] = []
    camera_images_dir = images_dir / camera_alias
    role_dir = camera_images_dir / "role_masks"
    if save_role_masks:
        role_dir.mkdir(parents=True, exist_ok=True)

    if role_workers > 1:
        if role_executor is None or segmenter_pool is None:
            raise ValueError("role_workers > 1 requires role_executor and segmenter_pool")
        futures = {role_executor.submit(segment_role_from_pool, segmenter_pool, image, role): idx for idx, role in enumerate(roles)}
        role_results: list[RoleResult | None] = [None] * len(roles)
        for future in as_completed(futures):
            role_results[futures[future]] = future.result()
        results = [result for result in role_results if result is not None]
    else:
        results = [segmenter.segment_role(image, role) for role in roles]

    for role, result in zip(roles, results, strict=False):
        role_mask = result.mask
        keep |= role_mask
        if save_role_masks:
            Image.fromarray(role_mask.astype(np.uint8) * 255).save(role_dir / f"{frame_stem}_{result.name}.png")
        role_logs.append(
            {
                "name": result.name,
                "prompts": result.prompts,
                "scores": result.scores,
                "boxes_xyxy": result.boxes_xyxy,
                "search_regions": role.get("search_regions", []),
                "confidence_threshold": result.confidence_threshold,
                "elapsed_ms": round(result.elapsed_ms, 3),
                "missing": result.missing,
                "optional": bool(role.get("optional", False)),
                "error": result.error,
            }
        )

    dilation_px = int(sam_cfg.get("mask_dilation_px", 0))
    keep = dilate_mask(keep, dilation_px)
    camera_images_dir.mkdir(parents=True, exist_ok=True)
    image_kind_set = set(image_kinds)
    outputs: dict[str, str] = {}

    if "filtered" in image_kind_set:
        filtered = apply_keep_mask(image, keep, int(sam_cfg.get("background_value", 0)))
        path = camera_images_dir / f"{frame_stem}_filtered.png"
        Image.fromarray(filtered).save(path)
        outputs["filtered"] = str(path.relative_to(out_dir))
    if "overlay" in image_kind_set:
        overlay = make_overlay(image, keep)
        path = camera_images_dir / f"{frame_stem}_overlay.png"
        Image.fromarray(overlay).save(path)
        outputs["overlay"] = str(path.relative_to(out_dir))
    if "keep_mask" in image_kind_set:
        path = camera_images_dir / f"{frame_stem}_keep_mask.png"
        Image.fromarray(keep.astype(np.uint8) * 255).save(path)
        outputs["keep_mask"] = str(path.relative_to(out_dir))
    if "config_boxes" in image_kind_set:
        boxes_preview = draw_config_boxes(image, roles)
        path = camera_images_dir / f"{frame_stem}_config_boxes.png"
        Image.fromarray(boxes_preview).save(path)
        outputs["config_boxes"] = str(path.relative_to(out_dir))

    return {
        "camera": camera_alias,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "keep_coverage": float(keep.mean()),
        "roles": role_logs,
        "outputs": outputs,
    }


def process_camera_from_pool(
    segmenter_pool: ThreadLocalSegmenterPool,
    image: np.ndarray,
    camera_alias: str,
    camera_cfg: dict[str, Any],
    sam_cfg: dict[str, Any],
    out_dir: Path,
    images_dir: Path,
    frame_stem: str,
    image_kinds: list[str],
    save_role_masks: bool,
    role_workers: int = 1,
    role_executor: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
    return process_camera(
        image=image,
        camera_alias=camera_alias,
        camera_cfg=camera_cfg,
        segmenter=segmenter_pool.get() if role_workers == 1 else None,
        sam_cfg=sam_cfg,
        out_dir=out_dir,
        images_dir=images_dir,
        frame_stem=frame_stem,
        image_kinds=image_kinds,
        save_role_masks=save_role_masks,
        role_workers=role_workers,
        role_executor=role_executor,
        segmenter_pool=segmenter_pool,
    )


def get_dataset_fps(dataset: Any) -> float | None:
    fps = getattr(getattr(dataset, "meta", None), "fps", None)
    if fps is None:
        return None
    try:
        fps_f = float(fps)
    except (TypeError, ValueError):
        return None
    return fps_f if fps_f > 0 else None


def write_mp4_from_images(image_paths: list[Path], output_path: Path, fps: float, codec: str = "mp4v") -> dict[str, Any]:
    import cv2  # type: ignore

    if not image_paths:
        return {"path": str(output_path), "frames": 0, "skipped": True}

    first = np.asarray(Image.open(image_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")
    try:
        for image_path in image_paths:
            frame = np.asarray(Image.open(image_path).convert("RGB"))
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return {"path": str(output_path), "frames": len(image_paths), "fps": fps, "codec": codec}


def write_output_videos(
    out_dir: Path,
    images_dir: Path,
    camera_aliases: list[str],
    video_kinds: list[str],
    fps: float,
    codec: str,
) -> list[dict[str, Any]]:
    video_outputs: list[dict[str, Any]] = []
    videos_dir = out_dir / "videos"
    for camera_alias in camera_aliases:
        for kind in video_kinds:
            camera_images_dir = images_dir / camera_alias
            image_paths = sorted(camera_images_dir.glob(f"ep*_frame*_{kind}.png"))
            if not image_paths:
                image_paths = sorted(images_dir.glob(f"ep*_frame*_{camera_alias}_{kind}.png"))
            if not image_paths:
                continue
            output_path = videos_dir / f"{camera_alias}_{kind}.mp4"
            try:
                result = write_mp4_from_images(image_paths, output_path, fps=fps, codec=codec)
            except Exception as exc:
                result = {
                    "path": str(output_path),
                    "camera": camera_alias,
                    "kind": kind,
                    "error": repr(exc),
                }
            else:
                result["camera"] = camera_alias
                result["kind"] = kind
            video_outputs.append(result)
    return video_outputs


def print_completion_summary(summary: dict[str, Any]) -> None:
    timing = summary.get("timing", {})
    frame_avg = (timing.get("frame_wall_ms") or {}).get("avg_ms")
    camera_avg = (timing.get("camera_wall_ms") or {}).get("avg_ms")
    print(
        "timing summary: "
        f"camera_workers={summary.get('camera_workers')} "
        f"role_workers={summary.get('role_workers')} "
        f"avg_frame_wall_ms={frame_avg} "
        f"avg_camera_image_ms={camera_avg}"
    )
    videos = summary.get("videos", [])
    if videos:
        print("videos written:")
        for video in videos:
            if "error" in video:
                print(f"  ERROR {video.get('camera')} {video.get('kind')}: {video.get('error')}")
            else:
                print(f"  {video.get('path')} ({video.get('frames')} frames @ {video.get('fps')} fps)")


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
    if args.frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    image_kinds = parse_csv(args.image_kinds)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        return_uint8=True,
        download_videos=not args.no_download_videos,
    )
    camera_keys = ds_cfg.get("camera_keys", {})
    default_confidence_threshold, processor_confidence_threshold = config_confidence_thresholds(config)
    camera_workers = max(1, int(args.camera_workers))
    role_workers = max(1, int(args.role_workers))
    use_segmenter_pool = camera_workers > 1 or role_workers > 1
    segmenter = build_segmenter(config, mock=args.mock, mock_text_mask=args.mock_text_mask) if not use_segmenter_pool else None
    segmenter_pool = ThreadLocalSegmenterPool(config, mock=args.mock, mock_text_mask=args.mock_text_mask) if use_segmenter_pool else None
    executor = ThreadPoolExecutor(max_workers=camera_workers) if camera_workers > 1 else None
    role_executor = ThreadPoolExecutor(max_workers=camera_workers * role_workers) if role_workers > 1 else None
    if camera_workers > 1:
        print(
            f"camera_workers={camera_workers}: processing cameras in parallel; "
            "each active worker loads its own SAM3 model."
        )
    if role_workers > 1:
        print(
            f"role_workers={role_workers}: processing roles in parallel; "
            f"up to {camera_workers * role_workers} SAM3 models may be loaded."
        )

    manifest_path = out_dir / "manifest.jsonl"
    dataset_fps = get_dataset_fps(dataset)
    video_fps = float(args.video_fps) if args.video_fps is not None else (dataset_fps or 15.0) / max(1, int(args.frame_stride))
    summary = {
        "repo_id": repo_id,
        "root": str(root) if root else None,
        "episodes": episodes,
        "config": str(args.config),
        "mock": args.mock,
        "frames_processed": 0,
        "camera_keys": camera_keys,
        "default_confidence_threshold": default_confidence_threshold,
        "processor_confidence_threshold": processor_confidence_threshold,
        "camera_workers": camera_workers,
        "role_workers": role_workers,
        "image_kinds": image_kinds,
        "dataset_fps": dataset_fps,
        "video_fps": video_fps,
    }

    per_episode_counts: dict[int, int] = {}
    timings = TimingAccumulator()
    try:
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

                frame_start = time.perf_counter()
                camera_entries: list[dict[str, Any] | None] = []
                camera_jobs: list[tuple[int, str, dict[str, Any], str, np.ndarray]] = []
                for camera_alias, camera_cfg in config.get("cameras", {}).items():
                    dataset_key = camera_keys.get(camera_alias, camera_cfg.get("dataset_key"))
                    if not dataset_key:
                        camera_entries.append({"camera": camera_alias, "error": "missing dataset_key"})
                        continue
                    if dataset_key not in item:
                        camera_entries.append(
                            {"camera": camera_alias, "dataset_key": dataset_key, "error": "key not present in dataset item"}
                        )
                        continue
                    image = tensor_or_array_to_uint8_hwc(item[dataset_key])
                    camera_entries.append(None)
                    camera_jobs.append((len(camera_entries) - 1, camera_alias, camera_cfg, dataset_key, image))

                if camera_workers == 1:
                    if role_workers == 1:
                        assert segmenter is not None
                    else:
                        assert segmenter_pool is not None
                        assert role_executor is not None
                    for job_idx, camera_alias, camera_cfg, dataset_key, image in camera_jobs:
                        cam_log = process_camera(
                            image=image,
                            camera_alias=camera_alias,
                            camera_cfg=camera_cfg,
                            segmenter=segmenter,
                            sam_cfg=config.get("sam3", {}),
                            out_dir=out_dir,
                            images_dir=images_dir,
                            frame_stem=frame_stem,
                            image_kinds=image_kinds,
                            save_role_masks=args.save_role_masks,
                            role_workers=role_workers,
                            role_executor=role_executor,
                            segmenter_pool=segmenter_pool,
                        )
                        cam_log["dataset_key"] = dataset_key
                        camera_entries[job_idx] = cam_log
                else:
                    assert executor is not None
                    assert segmenter_pool is not None
                    futures = {}
                    for job_idx, camera_alias, camera_cfg, dataset_key, image in camera_jobs:
                        future = executor.submit(
                            process_camera_from_pool,
                            segmenter_pool,
                            image,
                            camera_alias,
                            camera_cfg,
                            config.get("sam3", {}),
                            out_dir,
                            images_dir,
                            frame_stem,
                            image_kinds,
                            args.save_role_masks,
                            role_workers,
                            role_executor,
                        )
                        futures[future] = (job_idx, camera_alias, dataset_key)
                    for future in as_completed(futures):
                        job_idx, camera_alias, dataset_key = futures[future]
                        try:
                            cam_log = future.result()
                            cam_log["dataset_key"] = dataset_key
                        except Exception as exc:
                            cam_log = {
                                "camera": camera_alias,
                                "dataset_key": dataset_key,
                                "error": repr(exc),
                            }
                        camera_entries[job_idx] = cam_log

                frame_log["cameras"] = [entry for entry in camera_entries if entry is not None]
                frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000
                frame_log["elapsed_ms"] = round(frame_elapsed_ms, 3)
                timings.add_frame(frame_elapsed_ms)
                for cam_log in frame_log["cameras"]:
                    timings.add_camera_log(cam_log)

                manifest.write(json.dumps(frame_log, ensure_ascii=False) + "\n")
                summary["frames_processed"] += 1
                if args.max_total_frames is not None and summary["frames_processed"] >= args.max_total_frames:
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if role_executor is not None:
            role_executor.shutdown(wait=True)

    video_kinds = parse_csv(args.video_kinds)
    video_outputs: list[dict[str, Any]] = []
    if args.write_videos and video_kinds:
        video_outputs = write_output_videos(
            out_dir=out_dir,
            images_dir=images_dir,
            camera_aliases=list(config.get("cameras", {}).keys()),
            video_kinds=video_kinds,
            fps=video_fps,
            codec=str(args.video_codec),
        )

    summary["timing"] = timings.to_summary(camera_workers=camera_workers, role_workers=role_workers)
    summary["videos"] = video_outputs
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_completion_summary(summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to SAM3 filtering JSON config")
    parser.add_argument("--repo-id", default=None, help="Override config.dataset.repo_id")
    parser.add_argument("--root", default=None, help="Override config.dataset.root for local LeRobot dataset")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices, e.g. 0,1,2")
    parser.add_argument("--output-dir", default="examples/sam3_filtering/outputs/run", help="Directory for SAM3 filtering outputs")
    parser.add_argument(
        "--max-frames-per-episode",
        type=parse_optional_frame_limit,
        default=20,
        help="Limit frames processed per episode. Use 'all' or 0 for full episodes. Default 20 is for quick prompt/BBOX smoke tests.",
    )
    parser.add_argument(
        "--max-total-frames",
        type=parse_optional_frame_limit,
        default=None,
        help="Global frame limit. Use 'all' or 0 for no global limit.",
    )
    parser.add_argument("--frame-stride", type=int, default=15, help="Process every Nth frame within each selected episode")
    parser.add_argument(
        "--camera-workers",
        type=int,
        default=1,
        help="Number of cameras to process in parallel. Values >1 load one SAM3 model per active worker.",
    )
    parser.add_argument(
        "--role-workers",
        type=int,
        default=1,
        help="Number of roles to process in parallel per frame. This can multiply SAM3 model VRAM use.",
    )
    parser.add_argument("--no-download-videos", action="store_true", help="Do not download missing dataset videos")
    parser.add_argument("--mock", action="store_true", help="Use BBOX/dummy masks instead of importing/running SAM3")
    parser.add_argument(
        "--mock-text-mask",
        choices=["blank", "center"],
        default="center",
        help="Mock behavior for text-only roles without boxes",
    )
    parser.add_argument("--save-role-masks", action="store_true", help="Save individual role masks in addition to union masks")
    parser.add_argument(
        "--image-kinds",
        default="filtered",
        help="Comma-separated still image kinds to save under output-dir/images. Default saves only masked RGB images. Options: filtered,overlay,keep_mask,config_boxes.",
    )
    parser.add_argument(
        "--write-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write mp4 videos from saved per-camera image outputs.",
    )
    parser.add_argument(
        "--video-kinds",
        default="filtered",
        help="Comma-separated saved image kinds to encode as mp4, e.g. filtered,overlay,keep_mask,config_boxes.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output mp4 FPS. Default preserves dataset time after frame-stride: dataset_fps / frame_stride.",
    )
    parser.add_argument("--video-codec", default="mp4v", help="OpenCV fourcc codec for mp4 output")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
