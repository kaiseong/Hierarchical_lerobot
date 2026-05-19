# Frame 107 SAM3 Discrepancy Report

## Question

Why did standalone SAM3 on `front_107.jpg` detect the front-camera `robot`
prompt, while episode-level filtering for episode 0 frame 107 produced
`scores: []` for `front.robot_arm_gripper`?

## Current Conclusion

The only remaining supported cause is an input-image difference:

- Standalone success used `front_107.jpg`.
- Episode filtering uses the raw decoded LeRobot frame, saved locally as
  `front_107_raw.png`.
- These images are the same scene and size, but not the same pixels because
  `front_107.jpg` is JPEG-compressed.
- The `robot` prompt is unstable near the low threshold (`0.1`), so JPEG
  compression artifacts can create detections that are absent on the raw frame.

This is not fully closed until SAM3 is run on `front_107_raw.png` on a CUDA
machine.

## Evidence

### Same Source Video

`/home/kgs/sam_foundation_pose/front.mp4` and the LeRobot cached front video are
byte-identical:

```text
6af138f293edf8e7bc8a3eae42e8b74f67843045d3b975b5e10c969465f79cd8
```

This rules out a different source video.

### Same Episode Frame

LeRobot episode 0 item 107 reports:

```text
frame_index 107
timestamp 7.1333
```

The saved `front_107_raw.png` is pixel-identical to the LeRobot frame read by
the episode filter:

```text
mean_abs_diff 0.0
max_abs_diff 0
nonzero_pct 0.0
```

This rules out frame-index mismatch and raw-frame export mismatch.

### Decoder Difference Ruled Out

Earlier checks found ffmpeg frame `n=107`, LeRobot `torchcodec`, and LeRobot
`pyav` decode paths produced the same pixels for this frame.

This rules out torchcodec-vs-pyav-vs-ffmpeg decode drift.

### JPEG Diff Is Real

`front_107.jpg` differs from `front_107_raw.png`:

```text
mean_abs_diff 1.170892
max_abs_diff 28
nonzero_pct 71.939%
```

Re-encoding `front_107_raw.png` to JPEG with PIL quality 94 produces a close
image family:

```text
raw_png -> re-JPEG quality 94 vs front_107.jpg
mean_abs_diff 1.20085
max_abs_diff 13
nonzero_pct 72.07%
```

This supports that `front_107.jpg` is the same raw frame after lossy JPEG
encoding, not a different frame.

### Metadata And Color Path

Both images are RGB, 640x480, and have no EXIF rotation:

```text
front_107.jpg:     JPEG, RGB, 640x480, EXIF length 0
front_107_raw.png: PNG,  RGB, 640x480, EXIF length 0
```

The standalone script feeds SAM3 with:

```python
Image.open(image_path).convert("RGB")
```

The episode filter feeds SAM3 with:

```python
Image.fromarray(image)
```

The OpenCV BGR conversion in the standalone script is only for overlay
rendering, not for SAM3 inference.

This rules out RGB/BGR channel reversal, EXIF rotation, and image-size mismatch.

### Threshold And Postprocessing

The episode manifest for frame 107 recorded:

```text
front.robot_arm_gripper
prompts: ["robot"]
confidence_threshold: 0.1
scores: []
missing: true
```

Because `scores` is empty, this is not a `max_instances` truncation or mask
postprocessing removal. The processor returned no surviving robot candidates
for the raw episode input.

## Remaining CUDA Check

Run this on the CUDA machine:

```bash
cd ~/sam_foundation_pose
conda activate sam

python sam3_show_image_segment.py \
  --image front_107_raw.png \
  --prompt robot \
  --output front_107_raw_robot.png \
  --precision bf16 \
  --threshold 0.1 \
  --no-window
```

Expected confirming result:

```text
'robot': 0 mask(s), scores=[]
```

If this result appears, the discrepancy is confirmed to be caused by
`front_107.jpg` JPEG compression versus the raw episode frame.

If this command detects masks on `front_107_raw.png`, then the input-difference
hypothesis is wrong and the next suspect is a SAM3 call-path difference between
`sam3_show_image_segment.py` and `sam3_filter_dataset.py`.

## Next Branch If Raw PNG Still Detects

Run:

```bash
cd ~/Hierarchical_lerobot
conda activate sam

python examples/sam3_filtering/debug_frame_discrepancy.py \
  --image ~/sam_foundation_pose/front_107.jpg \
  --prompt robot \
  --threshold 0.1 \
  --precision bf16 \
  --repo-id rainbowrobotics/simtos_0412 \
  --episode 0 \
  --frame-index 107 \
  --camera-key observation.images.front \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --camera-alias front \
  --role-name robot_arm_gripper
```

Then compare direct standalone, direct dataset array, and filter call-path
outputs.
