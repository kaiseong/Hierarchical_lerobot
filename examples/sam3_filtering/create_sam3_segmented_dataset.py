#!/usr/bin/env python3
"""Create and upload a LeRobot dataset with SAM3-filtered camera observations.

The script reads an existing LeRobot dataset, applies the camera/role rules from
the SAM3 JSON config, writes a new LeRobot dataset with the same non-visual
features, and optionally uploads it to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam3_filter_dataset_v2 import (
    TimingAccumulator,
    apply_keep_mask,
    build_segmenter,
    config_confidence_thresholds,
    dilate_mask,
    load_json,
    parse_episodes,
    parse_optional_frame_limit,
    role_confidence_threshold,
    scalar_to_int,
    tensor_or_array_to_uint8_hwc,
)  # noqa: E402


def clone_features_for_new_dataset(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Copy source features while letting the new video encoder write fresh codec info."""
    cloned = copy.deepcopy(features)
    for feature in cloned.values():
        if feature.get("dtype") == "video":
            feature.pop("info", None)
    return cloned


def normalize_array_feature(value: Any, feature: dict[str, Any]) -> np.ndarray:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    dtype = np.dtype(feature["dtype"])
    expected_shape = tuple(feature["shape"])
    if arr.shape != expected_shape:
        if arr.shape == () and expected_shape == (1,):
            arr = arr.reshape(expected_shape)
        elif arr.size == int(np.prod(expected_shape)):
            arr = arr.reshape(expected_shape)
    return arr.astype(dtype, copy=False)


def normalize_writer_value(value: Any, feature: dict[str, Any]) -> Any:
    dtype = feature["dtype"]
    if dtype in {"image", "video"}:
        return tensor_or_array_to_uint8_hwc(value)
    if dtype == "string":
        return str(value)
    return normalize_array_feature(value, feature)


def role_result_to_log(role: dict[str, Any], result: Any, default_threshold: float) -> dict[str, Any]:
    return {
        "name": result.name,
        "prompts": result.prompts,
        "scores": result.scores,
        "boxes_xyxy": result.boxes_xyxy,
        "search_regions": role.get("search_regions", []),
        "confidence_threshold": result.confidence_threshold
        if result.confidence_threshold is not None
        else role_confidence_threshold(role, default_threshold),
        "elapsed_ms": round(result.elapsed_ms, 3),
        "missing": result.missing,
        "optional": bool(role.get("optional", False)),
        "error": result.error,
    }


def segment_camera_image(
    image: np.ndarray,
    camera_alias: str,
    camera_cfg: dict[str, Any],
    segmenter: Any,
    sam_cfg: dict[str, Any],
    default_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    roles = camera_cfg.get("roles", []) or []
    if not roles:
        raise ValueError(f"Camera '{camera_alias}' has no roles in the SAM3 config.")

    if hasattr(segmenter, "segment_roles"):
        results = segmenter.segment_roles(image, roles)
    else:
        results = [segmenter.segment_role(image, role) for role in roles]

    keep = np.zeros(image.shape[:2], dtype=bool)
    role_logs: list[dict[str, Any]] = []
    missing_required: list[str] = []
    errored_roles: list[str] = []

    for role, result in zip(roles, results, strict=False):
        keep |= result.mask
        if result.error:
            errored_roles.append(result.name)
        if result.missing and not bool(role.get("optional", False)):
            missing_required.append(result.name)
        role_logs.append(role_result_to_log(role, result, default_threshold))

    if bool(sam_cfg.get("fail_on_required_missing", False)) and missing_required:
        raise RuntimeError(
            f"Camera '{camera_alias}' is missing required SAM3 roles: {', '.join(missing_required)}"
        )
    if bool(sam_cfg.get("fail_on_role_error", False)) and errored_roles:
        raise RuntimeError(f"Camera '{camera_alias}' had SAM3 role errors: {', '.join(errored_roles)}")

    keep = dilate_mask(keep, int(sam_cfg.get("mask_dilation_px", 0)))
    filtered = apply_keep_mask(image, keep, int(sam_cfg.get("background_value", 0)))
    feature_ms = getattr(segmenter, "last_image_feature_ms", None)
    camera_log = {
        "camera": camera_alias,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "image_feature_ms": round(float(feature_ms), 3) if isinstance(feature_ms, (int, float)) else None,
        "keep_coverage": float(keep.mean()),
        "roles": role_logs,
        "missing_required": missing_required,
        "errored_roles": errored_roles,
    }
    return filtered, camera_log


def build_camera_mapping(config: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    ds_cfg = config.get("dataset", {})
    camera_keys = ds_cfg.get("camera_keys", {}) or {}
    mapping: dict[str, tuple[str, dict[str, Any]]] = {}
    for camera_alias, camera_cfg in (config.get("cameras", {}) or {}).items():
        dataset_key = camera_keys.get(camera_alias, camera_cfg.get("dataset_key"))
        if not dataset_key:
            raise ValueError(
                f"Camera '{camera_alias}' needs dataset.camera_keys['{camera_alias}'] or camera.dataset_key."
            )
        if dataset_key in mapping:
            prev_alias = mapping[dataset_key][0]
            raise ValueError(
                f"Dataset key '{dataset_key}' is mapped by both '{prev_alias}' and '{camera_alias}'."
            )
        mapping[str(dataset_key)] = (camera_alias, camera_cfg)
    return mapping


def build_writer_frame(
    item: dict[str, Any],
    source_features: dict[str, dict[str, Any]],
    default_feature_keys: set[str],
    filtered_images: dict[str, np.ndarray],
) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": str(item["task"])}
    for key, feature in source_features.items():
        if key in default_feature_keys:
            continue
        if key in filtered_images:
            frame[key] = filtered_images[key]
        else:
            frame[key] = normalize_writer_value(item[key], feature)
    return frame


def copy_optional_metadata(source_root: Path, output_root: Path) -> None:
    """Preserve optional metadata files that the writer does not regenerate."""
    for rel_path in ("meta/subtasks.parquet",):
        src = source_root / rel_path
        if src.exists():
            dst = output_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

    if args.repo_id == args.new_repo_id:
        raise ValueError("repo_id and new_repo_id must be different.")
    if args.frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}")

    config = load_json(args.config)
    if args.episodes is not None:
        episodes = parse_episodes(args.episodes)
    elif args.use_config_episodes:
        episodes = config.get("dataset", {}).get("episodes")
    else:
        episodes = None
    if episodes is not None:
        episodes = [int(e) for e in episodes]

    source_dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=episodes,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        return_uint8=True,
        download_videos=not args.no_download_videos,
    )

    output_root = Path(args.new_root) if args.new_root else HF_LEROBOT_HOME / args.new_repo_id
    if output_root.resolve() == Path(source_dataset.root).resolve():
        raise ValueError(f"Output root would overwrite the source dataset root: {output_root}")
    if output_root.exists():
        if not args.overwrite_local:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite-local to replace it."
            )
        shutil.rmtree(output_root)

    output_features = clone_features_for_new_dataset(source_dataset.meta.features)
    use_videos = any(feature.get("dtype") == "video" for feature in output_features.values())
    output_dataset = LeRobotDataset.create(
        repo_id=args.new_repo_id,
        fps=source_dataset.meta.fps,
        features=output_features,
        root=output_root,
        robot_type=source_dataset.meta.robot_type,
        use_videos=use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        encoder_threads=args.encoder_threads,
    )

    camera_mapping = build_camera_mapping(config)
    missing_configured_keys = sorted(set(camera_mapping) - set(source_dataset.meta.camera_keys))
    if missing_configured_keys and not args.allow_missing_cameras:
        raise KeyError(f"Configured camera keys are not in the source dataset: {missing_configured_keys}")

    default_threshold, processor_threshold = config_confidence_thresholds(config)
    segmenter = build_segmenter(config, mock=args.mock, mock_text_mask=args.mock_text_mask)
    sam_cfg = config.get("sam3", {}) or {}
    default_feature_keys = set(DEFAULT_FEATURES)
    timings = TimingAccumulator()

    manifest_path = output_root / "sam3_segmentation_manifest.jsonl"
    frames_written = 0
    source_frames_seen_by_episode: dict[int, int] = {}
    output_episodes = 0
    current_source_episode: int | None = None
    current_episode_has_frames = False
    start_time = time.perf_counter()

    def save_current_episode() -> None:
        nonlocal current_episode_has_frames, output_episodes
        if not current_episode_has_frames:
            return
        output_dataset.save_episode(parallel_encoding=not args.no_parallel_encoding)
        output_episodes += 1
        current_episode_has_frames = False
        if args.progress_every > 0:
            print(f"saved output episode {output_episodes - 1}")

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for rel_idx in range(len(source_dataset)):
                item = source_dataset[rel_idx]
                ep_idx = scalar_to_int(item.get("episode_index", 0))
                count = source_frames_seen_by_episode.get(ep_idx, 0)

                if current_source_episode is None:
                    current_source_episode = ep_idx
                elif ep_idx != current_source_episode:
                    save_current_episode()
                    current_source_episode = ep_idx

                source_frames_seen_by_episode[ep_idx] = count + 1
                if args.max_frames_per_episode is not None and count >= args.max_frames_per_episode:
                    continue
                if count % args.frame_stride != 0:
                    continue

                frame_idx = scalar_to_int(item.get("frame_index", item.get("index", rel_idx)))
                filtered_images: dict[str, np.ndarray] = {}
                frame_log = {
                    "source_relative_index": rel_idx,
                    "source_episode_index": ep_idx,
                    "source_frame_index": frame_idx,
                    "output_episode_index": output_episodes,
                    "output_frame_index": output_dataset.writer.episode_buffer["size"],
                    "task": str(item.get("task", "")),
                    "cameras": [],
                }
                frame_start = time.perf_counter()

                for dataset_key, (camera_alias, camera_cfg) in camera_mapping.items():
                    if dataset_key not in item:
                        if args.allow_missing_cameras:
                            frame_log["cameras"].append(
                                {
                                    "camera": camera_alias,
                                    "dataset_key": dataset_key,
                                    "error": "missing item key",
                                }
                            )
                            continue
                        raise KeyError(f"Dataset item is missing configured camera key '{dataset_key}'.")
                    image = tensor_or_array_to_uint8_hwc(item[dataset_key])
                    filtered, camera_log = segment_camera_image(
                        image=image,
                        camera_alias=camera_alias,
                        camera_cfg=camera_cfg,
                        segmenter=segmenter,
                        sam_cfg=sam_cfg,
                        default_threshold=default_threshold,
                    )
                    filtered_images[dataset_key] = filtered
                    camera_log["dataset_key"] = dataset_key
                    frame_log["cameras"].append(camera_log)
                    timings.add_camera_log(camera_log)

                writer_frame = build_writer_frame(
                    item=item,
                    source_features=source_dataset.meta.features,
                    default_feature_keys=default_feature_keys,
                    filtered_images=filtered_images,
                )
                output_dataset.add_frame(writer_frame)
                current_episode_has_frames = True
                frames_written += 1

                frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000
                timings.add_frame(frame_elapsed_ms)
                frame_log["elapsed_ms"] = round(frame_elapsed_ms, 3)
                manifest.write(json.dumps(frame_log, ensure_ascii=False) + "\n")

                if args.progress_every > 0 and frames_written % args.progress_every == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"processed {frames_written} frames "
                        f"({elapsed:.1f}s, source ep {ep_idx}, frame {frame_idx})"
                    )
                if args.max_total_frames is not None and frames_written >= args.max_total_frames:
                    break

        save_current_episode()
    finally:
        output_dataset.finalize()

    if frames_written == 0:
        raise RuntimeError(
            "No frames were written. Check --episodes, --max-frames-per-episode, and --frame-stride."
        )

    copy_optional_metadata(Path(source_dataset.root), Path(output_dataset.root))
    config_snapshot = output_root / "sam3_segmentation_config.json"
    shutil.copy2(args.config, config_snapshot)
    summary_path = output_root / "sam3_segmentation_summary.json"
    summary = {
        "source_repo_id": args.repo_id,
        "new_repo_id": args.new_repo_id,
        "source_root": str(source_dataset.root),
        "output_root": str(output_dataset.root),
        "config": str(args.config),
        "episodes": episodes,
        "frame_stride": args.frame_stride,
        "max_frames_per_episode": args.max_frames_per_episode,
        "max_total_frames": args.max_total_frames,
        "mock": args.mock,
        "frames_written": frames_written,
        "episodes_written": output_episodes,
        "camera_mapping": {key: alias for key, (alias, _) in camera_mapping.items()},
        "default_confidence_threshold": default_threshold,
        "processor_confidence_threshold": processor_threshold,
        "sam3": sam_cfg,
        "timing": timings.to_summary(camera_workers=1, role_workers=1),
        "manifest": str(manifest_path),
        "config_snapshot": str(config_snapshot),
        "elapsed_s": round(time.perf_counter() - start_time, 3),
        "push_requested": not args.no_push,
        "pushed": False,
    }
    write_json(summary_path, summary)

    if not args.no_push:
        output_dataset.push_to_hub(
            branch=args.branch,
            private=args.private,
            license=args.license,
            tag_version=not args.no_tag_version,
            upload_large_folder=args.upload_large_folder,
            tags=args.tags,
        )
        summary["pushed"] = True
        write_json(summary_path, summary)

        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo=summary_path.name,
            repo_id=args.new_repo_id,
            repo_type="dataset",
            revision=args.branch,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Source LeRobot dataset repo_id, e.g. user/source_dataset")
    parser.add_argument(
        "new_repo_id",
        help="Output Hugging Face dataset repo_id, e.g. user/source_dataset_sam3",
    )
    parser.add_argument(
        "--config",
        default="examples/sam3_filtering/configs/rby1_recycling_sam3_test.json",
        help="SAM3 filtering JSON config. The CLI repo_id overrides config.dataset.repo_id.",
    )
    parser.add_argument("--root", default=None, help="Optional local source dataset root")
    parser.add_argument("--new-root", default=None, help="Optional local output dataset root")
    parser.add_argument("--revision", default=None, help="Source dataset revision")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated source episode indices. Default processes all episodes.",
    )
    parser.add_argument(
        "--use-config-episodes",
        action="store_true",
        help="Use config.dataset.episodes when --episodes is not supplied.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=parse_optional_frame_limit,
        default=None,
        help="Limit source frames per episode. Use only for smoke tests; default writes full episodes.",
    )
    parser.add_argument(
        "--max-total-frames",
        type=parse_optional_frame_limit,
        default=None,
        help="Global frame limit for smoke tests.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame. Default 1.")
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
        help="Delete existing local output root first.",
    )
    parser.add_argument(
        "--allow-missing-cameras",
        action="store_true",
        help="Do not fail if a configured camera is absent.",
    )
    parser.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Do not download missing source videos.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Refresh source dataset cache from the Hub.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock masks instead of SAM3 for plumbing tests.",
    )
    parser.add_argument(
        "--mock-text-mask",
        default="center",
        choices=["center", "none"],
        help="Mock text-only mask mode.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Async image writer process count.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=0,
        help="Async image writer thread count.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Episodes per batch video encoding flush.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Global video encoder thread count.",
    )
    parser.add_argument(
        "--no-parallel-encoding",
        action="store_true",
        help="Encode per-camera episode videos sequentially.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Create the local dataset but skip Hugging Face upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload the target Hub dataset as private.",
    )
    parser.add_argument("--branch", default=None, help="Optional Hub branch for upload.")
    parser.add_argument("--license", default="apache-2.0", help="Dataset card license.")
    parser.add_argument(
        "--no-tag-version",
        action="store_true",
        help="Do not create/update the LeRobot version tag.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use HfApi.upload_large_folder for upload.",
    )
    parser.add_argument("--tags", nargs="*", default=["lerobot", "sam3-segmented"], help="Dataset card tags.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N written frames. 0 disables.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
