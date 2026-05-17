# Deep Interview Transcript Summary: rby1-vla-sam-foundationpose

- Created: 20260516T083604Z
- Profile: standard
- Final ambiguity: 19% <= threshold 20%
- Context snapshot: /home/kgs/.omx/context/rby1-vla-sam-foundationpose-20260516T081040Z.md

## Rounds

1. Initial task intake: RB-Y1 + LeRobot pi0.5 VLA demo robustness idea using SAM filtering and later FoundationPose/SE(3) conditioning.
2. Success direction: Maintain robust performance when inference environment differs from training data collection environment.
3. Domain shift scope: background, lighting, table/floor color, distractor objects, people appearing in camera; front camera fixed; wrist cameras move with wrists. Task is CAN/PET recycling with gray CAN bin and light-green PET bin. Consider left/right mirrored augmentation with camera swaps and state/action mirroring.
4. First milestone scope: image preprocessing only. Keep pi0.5 architecture and 16-axis state/action unchanged. Use same-resolution filtered RGB: keep target object, robot body/arms/grippers, and target bins; black out everything else.
5. Train/inference consistency: filtered RGB must be applied to both training and inference.
6. Runtime boundary: training data is SAM-filtered offline; inference applies SAM filtering online. Dataset collected at 15Hz. Inference target compute is Thor or RTX 5090.
7. Latency boundary: strict 15Hz SAM inference is not mandatory because pi0.5 uses current image/action chunks and RTC can cover some delay.
8. Success criteria: exploratory baseline comparison; user will judge after inference trials rather than predefine a fixed numeric threshold.

## Resolved Gates
- Non-goals: first milestone excludes pi0.5 architecture change, FoundationPose, object SE(3) conditioning, and EEF SE(3) state change.
- Decision boundaries: first milestone may only alter visual preprocessing; state/action remain 16-axis; training uses offline SAM-filtered dataset; inference uses online SAM filtering; latency is measured but not a hard 15Hz constraint.
- Pressure pass: confirmed that train/inference preprocessing mismatch is unacceptable; both must use same filtered image distribution.
