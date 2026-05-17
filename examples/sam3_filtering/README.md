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
