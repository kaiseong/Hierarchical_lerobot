# Handoff: RB-Y1 SAM3 Filtering Phase-0 Plan

## Repository

```text
https://github.com/kaiseong/Hierarchical_lerobot.git
branch: main
latest relevant commit: e1849d54 Support full-episode SAM3 consistency tests
```

## What this work is about

We are preparing a Phase-0 test for applying SAM3-based segmentation/filtering before a future VLA experiment on RB-Y1 with LeRobot pi0.5.

The future Phase-1 idea is:

```text
Original 3-camera RGB images
+ SAM3 masks for target object, robot arm/gripper, CAN/PET bins
-> same-resolution filtered RGB images
-> black background outside keep_mask
-> feed filtered RGB into pi0.5 without changing pi0.5 architecture/state/action
```

Phase-0 is **not training** and **not real-time demo integration**. Phase-0 only tests whether prompt/BBOX segmentation works on dataset frames and whether the filtered RGB images look usable.

## Robot / dataset assumptions

- Robot: RB-Y1
- VLA: LeRobot pi0.5
- Cameras: 3x RealSense D405
  - front/head
  - left wrist
  - right wrist
- Current state/action: dual arms + grippers only
  - 14 arm axes + 2 gripper axes = 16 axes
- Dataset frequency: 15Hz
- Task: recycling pick-and-place
  - CAN object -> gray CAN bin
  - PET/plastic bottle object -> light-green PET bin
- Training/inference distribution goal:
  - Train pi0.5 on SAM-filtered RGB dataset
  - In inference, server applies SAM filtering online before pi0.5 inference

## First milestone scope

In scope:

- Test SAM3 prompt/BBOX segmentation on LeRobot dataset frames.
- Generate filtered RGB images with same resolution as original.
- Keep pixels for:
  - target CAN/PET object
  - robot arm/gripper
  - gray CAN bin
  - light-green PET bin
- Fill all other pixels with black.
- Save overlay/mask/manifest outputs for human inspection.

Out of scope for first milestone:

- No pi0.5 architecture change.
- No extra mask channels or crop-token inputs.
- No FoundationPose.
- No object SE(3) flow-matching condition.
- No EEF SE(3) state replacement/addition.
- No final dataset converter yet.
- No real-time async server integration yet.

## Important design decisions

### Camera-specific SAM3 strategy

Front camera:

```text
PET/CAN object:
  text prompt

Gray CAN bin / light-green PET bin:
  shared fixed bin search-region BBOX + text prompt

Robot arm / gripper:
  dynamic BBOX + prompt eventually,
  broad workspace BBOX + prompt for Phase-0
```

Wrist cameras:

```text
PET/CAN object:
  text prompt

Gray/PET bins:
  text prompt only, optional, because bins may be out of view

Robot gripper:
  broad lower BBOX + prompt
```

### Why shared bin search regions?

Original thought was:

```text
gray CAN bin = fixed left BBOX
light-green PET bin = fixed right BBOX
```

But the training plan includes horizontal flip augmentation. After flip, gray/green bin positions are swapped. Therefore fixed role-specific bin BBOXes would become invalid.

Current implementation uses shared bin search regions:

```text
can_bin prompt = "gray bin"
pet_bin prompt = "light green bin"
shared search_regions = [left-bin-candidate, right-bin-candidate]
```

`search_regions` are not positive SAM visual prompts. They are post-processing filters: text-prompt masks are accepted only if they overlap the shared bin candidate areas.

### Full episode consistency

The data is already recorded at 15Hz.

- `--frame-stride 15` means sampling about 1 frame per second.
- For actual consistency check across the whole episode, use:

```text
--max-frames-per-episode all
--frame-stride 1
```

## Files to read first

```text
README.md
examples/sam3_filtering/README.md
examples/sam3_filtering/sam3_filter_dataset.py
examples/sam3_filtering/configs/rby1_recycling_sam3_test.json
```

OMX/deep-interview source artifacts on the original machine, if available:

```text
/home/kgs/.omx/specs/deep-interview-rby1-vla-sam-foundationpose.md
/home/kgs/.omx/interviews/rby1-vla-sam-foundationpose-20260516T083604Z.md
/home/kgs/.omx/context/rby1-vla-sam-foundationpose-20260516T081040Z.md
```

## How to run quick smoke test

Use this when first checking dataset keys, prompt/BBOX config, and output layout.

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --repo-id <your_dataset_repo_id> \
  --root <optional_local_dataset_root> \
  --episodes 0 \
  --max-frames-per-episode 20 \
  --frame-stride 15 \
  --save-role-masks \
  --output-dir examples/sam3_filtering/outputs/sam3_ep0_quick
```

## How to run full 15Hz episode consistency test

Use this after prompt/BBOX looks roughly correct.

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --repo-id <your_dataset_repo_id> \
  --root <optional_local_dataset_root> \
  --episodes 0 \
  --max-frames-per-episode all \
  --frame-stride 1 \
  --save-role-masks \
  --output-dir examples/sam3_filtering/outputs/sam3_ep0_full_15hz
```

## How to run mock mode on low-spec PC

Mock mode does not import/run SAM3. It validates dataset loading, BBOX config, and output plumbing.

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --mock \
  --repo-id <your_dataset_repo_id> \
  --root <optional_local_dataset_root> \
  --episodes 0 \
  --max-frames-per-episode 2 \
  --frame-stride 30 \
  --output-dir examples/sam3_filtering/outputs/mock_ep0
```

## Environment preference

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e ".[dataset,pi]"
```

For PyTorch on RTX 5090, use the official selector:

```text
https://docs.pytorch.org/get-started/locally/
```

SAM3 install:

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ../Hierarchical_lerobot
```

## What the next Codex should do

1. Pull the latest repo.
2. Read this handoff and the README files.
3. Confirm the real dataset `repo_id` or local root.
4. Confirm actual LeRobot camera keys.
5. Update `examples/sam3_filtering/configs/rby1_recycling_sam3_test.json`:
   - dataset repo/root
   - camera keys
   - front shared bin search regions
   - wrist gripper BBOXes
   - prompt candidates
6. Run mock mode if not on 5090/Thor.
7. On 5090/Thor, run quick SAM3 test.
8. Inspect:
   - `*_filtered.png`
   - `*_overlay.png`
   - `*_keep_mask.png`
   - `*_config_boxes.png`
   - `manifest.jsonl`
9. Adjust prompts/BBOX/dilation.
10. Run full 15Hz episode consistency test.

## Known caveats

- The current script is a Phase-0 inspection tool, not a final dataset converter.
- It currently uses SAM3 image segmentation, not video tracking.
- Video tracking may be evaluated later, but it requires stateful per-camera sessions.
- Dynamic BBOX from camera calibration/FK is not implemented yet.
- For bin flip augmentation, keep shared search regions unless the dataset explicitly marks whether a sample is flipped.
- Output PNGs can become large for full 15Hz episodes; use a scratch/output directory.
