# Execution Spec: RB-Y1 pi0.5 SAM-Filtered VLA Input

## Metadata
- Created: 20260516T083604Z
- Source: deep-interview
- Final ambiguity: 19%
- Threshold: 20%
- Context type: experimental VLA demo design
- Context snapshot: /home/kgs/.omx/context/rby1-vla-sam-foundationpose-20260516T081040Z.md

## Intent
Improve RB-Y1 pick-and-place/recycling demo robustness when inference happens in environments different from the training data collection environment, by reducing irrelevant visual background variation before images enter LeRobot pi0.5.

## Desired Outcome
Compare baseline pi0.5 against a SAM-filtered pi0.5 variant in shifted environments. The filtered variant should be less sensitive to background, lighting, table/floor color, distractor objects, and people appearing in the camera view.

## System Setup
- Robot: RB-Y1.
- Policy/model: LeRobot pi0.5.
- Cameras: 3x RealSense D405: front/head, right wrist, left wrist.
- Robot state/action: dual arms + grippers only; 14 arm axes + 2 gripper axes = 16 axes.
- Task: CAN/PET recycling pick-and-place.
- Bins: CAN bin is gray; PET bin is light green.
- Dataset collection rate: 15Hz.
- Inference compute target: Thor or RTX 5090.

## First Milestone Scope
Use SAM/SAM3-style visual filtering only. Preserve original image resolution. Pixels belonging to required roles retain original RGB values; all other pixels become black.

Required visual roles to keep:
1. Target object to pick: CAN or PET object depending on instruction/task.
2. Robot body/arms/grippers, especially visible arms and grippers.
3. Destination bins: CAN bin and PET bin, or at minimum the target bin plus any bin needed for disambiguation.

## Out of Scope / Non-goals for First Milestone
- No pi0.5 architecture change.
- No extra mask channels or crop-token inputs in first pass.
- No FoundationPose integration.
- No object SE(3) flow-matching condition.
- No EEF SE(3) robot-state replacement/addition.
- No fixed numeric success threshold before exploratory trials.

## Decision Boundaries
- Allowed: create offline SAM-filtered training dataset.
- Allowed: run online SAM filtering at inference time.
- Allowed: log latency and failure modes.
- Allowed: compare SAM prompt segmentation vs video tracking as implementation alternatives.
- Not allowed in first milestone without confirmation: changing pi0.5 input architecture or state/action dimensionality.
- Latency: strict 15Hz SAM filtering is not mandatory; action chunking and RTC may tolerate delay.

## Data Augmentation Idea
Mirror augmentation may be used:
- Left/right mirror camera images.
- Swap left/right wrist camera streams after mirroring.
- Mirror robot state and action axes appropriately.
- This can expose the model to swapped bin-side configurations even if real bin positions are fixed.

## Acceptance / Evaluation Criteria
Exploratory comparison rather than predefined threshold:
- Run baseline pi0.5 and SAM-filtered pi0.5 in shifted environments.
- Record success/failure and compare qualitatively/quantitatively after trials.
- Track failure types: grasp fail, wrong bin, place fail, distractor/background confusion, mask failure, latency/timing issue.
- Keep side-by-side videos and logs.

## Recommended Next Planning Focus
1. Select SAM mode for offline dataset generation and online inference.
2. Define prompt strategy for target object, bins, robot arms/grippers.
3. Design filtered RGB generation and logging format.
4. Define baseline vs filtered training/evaluation runs.
5. Keep FoundationPose/SE(3) as second-phase roadmap only.
