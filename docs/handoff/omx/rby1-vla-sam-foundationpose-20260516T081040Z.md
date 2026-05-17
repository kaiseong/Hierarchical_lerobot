# Deep Interview Context Snapshot: rby1-vla-sam-foundationpose

- Created: 20260516T081040Z
- Task statement: Clarify and design an RB-Y1 VLA demo idea using LeRobot pi0.5, SAM/SAM3-style segmentation, and possibly FoundationPose/SE(3) conditioning.
- Desired outcome: A clear, execution-ready experimental plan/spec for improving robustness of pick-and-place demos under diverse backgrounds/environments.
- Stated solution:
  1. Use camera images + instruction to keep only target object, robot body/arms/grippers, and target container/bin in VLA image inputs.
  2. Later consider FoundationPose to estimate object SE(3) and feed it as an additional condition to pi0.5 flow matching.
  3. Consider replacing/augmenting robot state with EEF SE(3).
- Known setup:
  - Robot: RB-Y1.
  - VLA: LeRobot pi0.5.
  - Cameras: 3x RealSense D405: head/front, right wrist, left wrist.
  - Action/state axes: dual arms + grippers; 14 arm joints + 2 gripper axes = 16 axes.
- Intent hypothesis: Improve VLA robustness by reducing visual distraction/background variation and adding more geometrically meaningful conditioning.
- Constraints: Need compatible with existing VLA demo pipeline; exact SAM mode and FoundationPose integration are unresolved.
- Unknowns/open questions:
  - Primary objective: demo robustness, novelty, success rate, generalization, or research contribution?
  - Runtime vs offline preprocessing requirement.
  - Whether to preserve full image context alongside filtered image/masks.
  - Which objects/roles must be segmented per camera.
  - What acceptance criteria define success.
  - What OMX/Codex may decide vs what requires user confirmation.
- Decision-boundary unknowns:
  - Whether first pass should be no-model-architecture-change preprocessing only.
  - Whether to fine-tune pi0.5 or only alter inference inputs.
  - Whether FoundationPose/SE(3) is in-scope for first milestone.
- Likely codebase touchpoints: LeRobot dataset/dataloader transforms, pi0.5 policy input adapters, camera preprocessing, inference/demo pipeline, config/eval scripts.
- Prompt-safe initial-context summary status: recorded
