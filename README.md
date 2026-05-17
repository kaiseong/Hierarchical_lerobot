# Hierarchical LeRobot — RB-Y1 SAM3 Filtering Phase 0

이 repository는 RB-Y1 + LeRobot pi0.5 데모에서 **SAM3 기반 이미지 전처리**를 1차 실험에 넣기 전에, 먼저 prompt/BBOX segmentation이 실제 데이터셋에서 잘 되는지 확인하기 위한 0차 테스트 코드를 담고 있다.

## 목표

최종 1차 목표는 기존 pi0.5 구조와 16축 state/action은 그대로 두고, VLA 입력 RGB 이미지만 다음처럼 바꾸는 것이다.

```text
original RGB
  + SAM3 keep mask(target object + robot arm/gripper + CAN/PET bins)
  -> same-resolution filtered RGB
  -> background pixels = black
  -> pi0.5 input
```

0차 목표는 아직 학습/실시간 데모를 하지 않고, **데이터셋 샘플 이미지에 SAM3를 적용해 어떤 prompt/BBOX 전략이 잘 되는지 확인**하는 것이다.

## 추가된 파일

```text
examples/sam3_filtering/
├── sam3_filter_dataset.py
├── configs/
│   └── rby1_recycling_sam3_test.json
└── outputs/                 # 실행 결과 저장 위치, git ignore 권장
```

- `sam3_filter_dataset.py`: LeRobot dataset을 읽고 camera별 SAM3 설정을 적용해 filtered/overlay/mask PNG와 manifest를 저장한다.
- `rby1_recycling_sam3_test.json`: front/right wrist/left wrist별 prompt와 BBOX 정책을 정의한다.

## 0차 실험 설계

### Front camera

Front pose는 고정이고 CAN/PET 통 위치도 고정이라고 가정한다. 단, 좌우 flip 증강을 하면 gray CAN bin과 light-green PET bin의 이미지상 위치가 서로 바뀌므로, 0차 테스트에서는 두 통 모두 같은 shared bin search-region BBOX 후보를 사용하고 text prompt로 색/종류를 구분한다.

| 대상 | 방식 |
|---|---|
| CAN/PET 물체 | text prompt |
| Gray CAN bin | shared fixed bin search-region BBOX + prompt |
| Light-green PET bin | shared fixed bin search-region BBOX + prompt |
| Robot arm / gripper | dynamic BBOX + prompt, 또는 broad workspace BBOX + prompt |

### Wrist cameras

Wrist는 팔 움직임에 따라 시야가 크게 바뀐다.

| 대상 | 방식 |
|---|---|
| CAN/PET 물체 | text prompt |
| Gray/PET bin | text prompt only, optional |
| Robot gripper | broad lower BBOX + prompt |

Wrist에서 bin은 안 보일 수 있으므로 optional로 둔다. Gripper BBOX는 mask가 잘리지 않도록 하단 넓은 영역에서 시작한다.

## JSON 설정 파일

기본 설정 파일:

```text
examples/sam3_filtering/configs/rby1_recycling_sam3_test.json
```

먼저 반드시 수정해야 할 부분:

```json
{
  "dataset": {
    "repo_id": "kaiseong/rby1-recycling-demo",
    "root": null,
    "episodes": [0],
    "camera_keys": {
      "front": "observation.images.front",
      "left_wrist": "observation.images.left_wrist",
      "right_wrist": "observation.images.right_wrist"
    }
  }
}
```

실제 데이터셋의 camera key가 다르면 `camera_keys`를 바꿔야 한다. 확인용 예시:

```bash
python examples/dataset/load_lerobot_dataset.py
```

또는 Python에서:

```python
from lerobot.datasets import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata("<your_dataset_repo_id>", root="<optional_local_root>")
print(meta.camera_keys)
print(meta.features)
```

### BBOX format

현재 config는 `normalized_xyxy`를 기본으로 쓴다.

```json
{
  "format": "normalized_xyxy",
  "value": [0.05, 0.35, 0.45, 0.95]
}
```

의미:

```text
[x0, y0, x1, y1]
0.0~1.0 normalized image coordinates
```

지원 format:

```text
normalized_xyxy
normalized_cxcywh
pixel_xyxy
pixel_xywh
```


### Flip 증강과 bin BBOX 주의점

Front에서 `can_bin`과 `pet_bin`에 서로 다른 role-specific positive BBOX를 고정하면, 좌우 flip 증강 이미지에서는 CAN/PET 위치가 반대로 바뀌어 BBOX가 틀어진다. 그래서 config는 두 bin role 모두에 같은 `search_regions`를 사용한다.

```text
can_bin prompt = "gray bin"
pet_bin prompt = "light green bin"
shared search_regions = [left-bin-candidate, right-bin-candidate]
```

`search_regions`는 SAM3에 positive object box로 넣는 `boxes`와 다르다. Text prompt로 나온 mask 후보 중 shared bin 영역과 겹치는 것만 채택하기 위한 후처리 filter다. 즉, flip 증강에 안전하게 양쪽 bin 후보 영역을 모두 허용하되, 실제 구분은 prompt가 한다.

## 설치

아래는 RTX 5090 또는 Thor 서버에서 실행하는 것을 기준으로 한다. 이 repo를 로컬 MX250 PC에서 실행하면 SAM3 실제 inference는 매우 느리거나 실패할 수 있다. 로컬 PC에서는 `--mock`으로 dataset/key/output plumbing만 확인한다.

### 1. Repo 준비

```bash
git clone https://github.com/kaiseong/Hierarchical_lerobot.git
cd Hierarchical_lerobot
```

이미 `/home/kgs/lerobot` 같은 로컬 checkout을 쓰는 경우에는 해당 경로에서 진행해도 된다.

### 2. Python 환경

Python 3.10 또는 3.11 환경을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 3. PyTorch 설치

RTX 5090은 Blackwell 계열이므로 PyTorch/CUDA wheel 호환성이 중요하다. 먼저 공식 PyTorch selector에서 현재 driver에 맞는 CUDA build를 확인한다.

- 공식 설치 selector: https://docs.pytorch.org/get-started/locally/

예시 형태:

```bash
# 예시일 뿐이다. 실제 5090 서버의 driver/CUDA에 맞춰 PyTorch 공식 selector 명령을 사용한다.
pip install torch torchvision torchaudio --index-url <official-pytorch-cuda-wheel-index>
```

설치 후 확인:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

### 4. LeRobot 설치

```bash
pip install -e ".[dataset,pi]"
```

필요하면 시각화/학습 extras를 추가한다.

```bash
pip install -e ".[dataset,pi,training,viz]"
```

### 5. SAM3 설치

공식 SAM3 repository를 같은 상위 폴더에 clone해서 editable 설치한다.

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ../Hierarchical_lerobot
```

SAM3 checkpoint가 Hugging Face에서 자동 다운로드되는 경우가 있으므로, 필요하면 로그인한다.

```bash
huggingface-cli login
```

### 6. 추가 유틸

```bash
pip install pillow opencv-python
```

`opencv-python`은 mask dilation에 사용된다. 없어도 PIL fallback을 사용한다.

## 사용법

### A. 로컬/저사양 PC에서 mock 실행

SAM3를 실제로 돌리지 않고, BBOX와 출력 구조만 확인한다.

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

### B. RTX 5090/Thor에서 SAM3 실제 실행

```bash
python examples/sam3_filtering/sam3_filter_dataset.py \
  --config examples/sam3_filtering/configs/rby1_recycling_sam3_test.json \
  --repo-id <your_dataset_repo_id> \
  --root <optional_local_dataset_root> \
  --episodes 0 \
  --max-frames-per-episode 20 \
  --frame-stride 15 \
  --save-role-masks \
  --output-dir examples/sam3_filtering/outputs/sam3_ep0
```

처음에는 episode 1개, frame 10~20장만 처리해서 prompt/BBOX가 맞는지 확인한다. 전체 dataset preprocessing은 아직 하지 않는다.

## 출력물

실행 후 output dir에는 다음이 저장된다.

```text
summary.json
manifest.jsonl
ep000000_frame000000_front_filtered.png
                    front_overlay.png
                    front_keep_mask.png
                    front_config_boxes.png
                    left_wrist_filtered.png
                    ...
role_masks/          # --save-role-masks 사용 시
```

- `*_filtered.png`: background가 black으로 지워진 최종 VLA 입력 후보.
- `*_overlay.png`: keep mask가 초록색으로 overlay된 검수 이미지.
- `*_keep_mask.png`: union keep mask.
- `*_config_boxes.png`: JSON에 정의한 BBOX를 원본 이미지 위에 그린 확인 이미지.
- `manifest.jsonl`: frame/camera/role별 score, missing 여부, 처리 시간.

## 확인 기준

0차에서 봐야 할 것:

1. Front에서 CAN/PET bin BBOX가 실제 통 전체와 rim/interaction zone을 포함하는가?
2. Wrist에서 하단 gripper BBOX가 gripper open/close 시에도 잘리지 않는가?
3. CAN/PET object text prompt가 실제 물체를 잡는가?
4. `PET` 단독 prompt 대신 `plastic bottle`, `PET bottle`이 더 안정적인가?
5. robot/gripper prompt가 배경 금속물, 케이블, fixture를 오탐하지 않는가?
6. filtered RGB에서 조작에 필요한 픽셀(object, gripper, bin rim)이 검게 지워지지 않는가?
7. mask dilation 5/12/20 px 중 어느 정도가 가장 안전한가?

## 0차 이후 1차 계획

0차에서 prompt/BBOX가 안정적이면 다음 단계로 간다.

```text
1. SAM3 config 확정
2. offline SAM-filtered training dataset 생성
3. pi0.5를 filtered RGB dataset으로 학습
4. server-side online SAM filtering + pi0.5 async inference 구성
5. baseline RGB pi0.5 vs SAM-filtered pi0.5 비교
```

1차 범위에서 제외:

```text
- pi0.5 architecture 변경
- mask channel/crop token 추가
- FoundationPose
- object SE(3) condition
- EEF SE(3) robot state 변경
```

## 현재 부족하거나 직접 채워야 하는 정보

아래는 5090에서 실행 전에 반드시 확인해야 한다.

1. 실제 LeRobot dataset repo_id 또는 local root.
2. 실제 camera key 이름.
3. Front camera 기준 CAN/PET bin BBOX.
4. Wrist camera 기준 gripper가 보이는 영역.
5. SAM3 설치/체크포인트 접근 권한.
6. RTX 5090 서버의 PyTorch/CUDA 호환성.
7. 결과를 보고 prompt 후보를 줄일지, bbox를 넓힐지, dilation을 조정할지 결정.

