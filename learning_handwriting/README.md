# 개인 손글씨 신경망 파이프라인

한 번 대량 학습한 뒤에는 **ChatGPT/Gemini에서 답변을 복사 → 로컬 앱의 붙여넣기 버튼 → PNG/PDF 다운로드**만 하는 프로젝트입니다. 글리프를 매번 수동 조판하는 방식이 아닙니다.

모델은 `JamoGlyphNet`입니다. 한글은 초성(19)·중성(21)·종성(28)을 각각 임베딩하여 MLP로 합친 다음 CNN 디코더가 한 음절의 잉크 마스크를 생성합니다. 따라서 학습에 없는 한글 음절도, 충분히 다양한 음절 조합으로 학습했다면 생성할 수 있습니다. 영어·숫자·기호는 문자 임베딩 경로를 씁니다. 한 사람의 글씨체를 복제하는 용도로 작고 이해하기 쉬운 CNN/MLP 모델입니다.

> 이 도구는 본인이 작성·검토할 수 있는 내용에 사용하세요. 과제의 AI 사용 및 필기 제출 규정을 따르고, 사실성·인용은 직접 확인해야 합니다.

## 0. 프로젝트 구조

```
data/raw/                 # 스캔한 원본
data/crops/               # 한 칸/한 글자 이미지
data/labels.csv           # path,char (정답 라벨)
checkpoints/              # GPU 학습 결과
handwriting/              # 모델·전처리·렌더러
tools/                    # 격자 분할/라벨 시트/OCR 보조
server.py                 # 학습 후 로컬 추론 UI
```

## 1. 데이터: 한 번만 대량으로 준비

가장 안정적인 방법은 **격자 필기지**입니다. `chars.txt`에 학습할 문자를 한 글자씩 넣고 필기지를 만듭니다.

```powershell
python tools/make_label_sheet.py --chars chars.txt --out data/raw/sheet.pdf
```

출력물을 인쇄해 각 칸에 해당 글자를 쓰고 300dpi 이상으로 스캔합니다. 스캔은 균일한 조명, 검정/파랑 펜, 흰 종이, 한 글자당 96px 이상이 좋습니다. 최소 3~5회 반복, 권장은 문자당 15~30회입니다.

스캔을 격자로 잘라 라벨을 자동 배정합니다.

```powershell
python tools/segment_grid.py --image data/raw/scan_01.png --chars chars.txt --rows 20 --cols 10 --out data/crops --csv data/labels.csv
```

잘린 결과를 반드시 빠르게 검수하세요. 기울거나 칸 위치가 틀리면 `--left --top --cell-w --cell-h`로 조정합니다. 자유문장 스캔의 텍스트 초안이 필요할 경우 PaddleOCR 보조 스크립트를 씁니다. OCR 결과는 정답이 아니므로 사람이 수정해야 합니다.

```powershell
pip install paddleocr
python tools/ocr_assist.py --image data/raw/freeform.png
```

### 문자 범위

- 한글: 자모 `ㄱ…ㅎ`, `ㅏ…ㅣ`와 실제 자주 쓸 완성형 음절을 폭넓게 수집합니다. 모든 11,172 음절을 쓸 필요는 없지만 조합을 많이 포함할수록 미등록 음절 품질이 좋습니다.
- 영어: `A-Z`, `a-z` 각각.
- 숫자: `0-9`.
- 기호: `. , : ; ! ? ' " ( ) [ ] { } + - × ÷ = < > / \\ % # @ & _` 및 과제에서 쓰는 기호.
- 공백과 줄바꿈은 생성하지 않고 레이아웃 엔진이 처리합니다.

`chars.txt`에는 한 줄에 한 글자만 넣습니다. 같은 문자를 여러 줄에 반복하면 여러 필기 샘플이 됩니다.

## 2. SSH GPU 서버에서 학습

로컬에서 데이터 폴더를 서버로 복사한 뒤, 서버에서 실행합니다.

```bash
git clone <이-프로젝트-저장소> handwriting
cd handwriting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0,2 python -m handwriting.train --csv data/labels.csv --epochs 120 --batch-size 128 --device cuda --out checkpoints/my_style.pt
```

`CUDA_VISIBLE_DEVICES=0,2`는 서버의 물리 GPU 0번과 2번만 이 작업에 보이게 합니다. 코드가 두 장 이상을 감지하면 단일 서버 `DataParallel`로 두 GPU를 모두 사용합니다. 학습 중 `checkpoints/samples/epoch_*.png`를 확인하세요. 선이 뭉개지면 해상도/정렬을 개선하고, 문자마다 샘플을 늘리는 편이 모델을 키우는 것보다 효과가 큽니다. 모델·가중치·추론 설정은 하나의 `.pt` 체크포인트에 저장됩니다.

## 3. 로컬에서 한 번 설정 후 사용

학습된 `checkpoints/my_style.pt`를 PC로 가져옵니다. CPU 추론도 충분합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py --checkpoint checkpoints/my_style.pt
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. ChatGPT의 **복사** 버튼을 누른 뒤 앱의 **클립보드 붙여넣기**를 누르고, 필요하면 원고를 편집한 후 다운로드합니다. 브라우저 보안상 클립보드는 사용자의 버튼 클릭 후에만 읽습니다.

출력은 PNG와 PDF입니다. 삼성 노트는 이를 가져와 주석은 추가할 수 있지만, 가져온 이미지/PDF의 잉크 획을 네이티브 필기처럼 지우는 것은 보장되지 않습니다. 획 단위 편집까지 필요하면 펜 좌표(stroke) 데이터로 별도 벡터 생성 모델을 학습해야 합니다.

## 공부 포인트

`handwriting/model.py`에서 다음을 순서대로 바꿔보며 실험할 수 있습니다.

1. 임베딩 차원과 MLP 깊이
2. 전치합성곱 CNN 디코더의 채널 수
3. BCE + L1 손실 가중치
4. 좌우 반전이 아닌 작은 회전/이동/두께 증강
5. 이후 조건부 VAE(잠재 노이즈)나 PatchGAN 판별기 추가

검증 손실이 낮아도 글씨가 자연스럽지 않을 수 있으니 샘플 이미지를 우선 평가하세요.
