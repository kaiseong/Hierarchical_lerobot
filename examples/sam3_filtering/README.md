# SAM3 Filtering Phase-0 Test

Root `README.md`에 설치/사용법 전체가 정리되어 있다.

Quick start:

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --mock \
  --repo-id <your_dataset_repo_id> \
  --episodes 0 \
  --max-frames-per-episode 2 \
  --output-dir examples/sam3_filtering/outputs/mock_ep0
```

RTX 5090/Thor에서 실제 SAM3 실행:

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --repo-id <your_dataset_repo_id> \
  --episodes 0 \
  --max-frames-per-episode 20 \
  --frame-stride 15 \
  --save-role-masks \
  --output-dir examples/sam3_filtering/outputs/sam3_ep0
```


### Quick test vs full-episode consistency test

데이터셋은 이미 15Hz로 녹화되어 있다. 따라서 `--frame-stride 15`는 전체 15Hz 처리가 아니라 **약 1초에 1장만 샘플링**하는 quick test 설정이다. Prompt/BBOX가 대략 맞는지 빠르게 볼 때만 사용한다.

실제 추론처럼 episode 전체에서 segmentation이 계속 일관적인지 보려면 다음처럼 실행한다.

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

권장 순서는:

```text
1. quick test: --max-frames-per-episode 20 --frame-stride 15
2. full consistency test: --max-frames-per-episode all --frame-stride 1
```

Full test는 15Hz 모든 프레임을 처리하므로 오래 걸리고 output PNG도 많이 생긴다.
