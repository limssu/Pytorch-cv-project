# 🔎딥러닝 기반 반도체 웨이퍼 결함 탐지 및 유형 분류 연구
> **Semiconductor Wafer Defect Detection and Classification using Deep Learning Features**

## 🎯 1. 프로젝트 개요 및 연구 동기 (Introduction)
- **주제**: 이미지 비전 센서 데이터를 기반으로 반도체 공정 중 발생하는 웨이퍼 결함을 실시간 탐지(Detection)하고 유형을 분류(Classification)함.
- **연구 동기**:  `2025 국제인공지능산업대전(AI EXPO KOREA)`,`2026 국제인공지능산업대전(AI EXPO KOREA)` 참관을 통해 대량 양산 및 미세 공정 최적화 흐름 속에서 AI 비전 검사 솔루션이 반도체 수율(Yield) 향상의 핵심 동력임을 직접 목격함. 이에 최신 컴퓨터 비전 방법론을 반도체 도메인에 적용하여 실무적 난제를 해결하고자 본 연구를 기획함.

---

## 🏗️ 2. 프로젝트 아키텍처 (Inspection System Architecture)
*프로젝트 수행 및 엔지니어링 진행 상황에 따라 기술 스택 및 파이프라인 구조는 유연하게 변경될 수 있습니다.*

- **컴퓨팅 인프라 (Environment)**: AWS EC2 (`g4dn.xlarge`) 인스턴스 + Ubuntu 24.04 + Miniconda 가상환경 (`anomaly_env`)
- **딥러닝 프레임워크 (Framework)**: PyTorch (`torch`, `torchvision`) 기반의 종단간(End-to-End) 모델링
- **데이터 파이프라인 (Data)**: 산업용 비전 표준 벤치마크 데이터셋 `MVTec AD` + 반도체 웨이퍼 맵 데이터 `WM-811K`
- **비전 전처리 엔진 (Preprocessing)**: OpenCV 기반 영상 대비 극대화 알고리즘 (**CLAHE**) 탑재
- **프로세스 관리 및 로깅 (M&E)**: 리눅스 백그라운드 프로세스 기반의 실시간 `train.log` 아카이빙 로직 구현

---

## 📊 3. 성능 정량 평가 기준 (Quantitative Evaluation Metrics)
현업 인라인(In-line) 자동 검사 시스템(ADC) 도입 가능성을 타당성 있게 검증하기 위해 학술적·실무적 관점에서 다음과 같은 명확한 평가 지표를 수립합니다.

### A. 픽셀 단위 결함 국소화(Localization) 능력 평가
- **IoU (Intersection Over Union)**: 가상 특징 공간 내 거리 분석을 통해 모델이 예측한 결함 영역(Predicted)과 실제 결함의 정답 영역(Ground-Truth)의 공간적 중첩도를 측정함.
- 본 연구에서는 $IoU \ge 0.5$ 이상을 유효 검출(True Positive)의 최소 기준으로 정의하여, 현업의 치명적 이슈인 오탐지율(False Positive Rate)을 최소화하는 최적의 임계값(Threshold)을 도출함.

### B. 다중 결함 검출 및 분류 성능 평가
- **mAP (Mean Average Precision) @50 및 @50:95**: 반도체 공정 특성상 불량의 클래스가 다양하므로(Scratch, Center, Edge 등), 각 결함 유형별 정밀도(Precision)와 재현율(Recall)을 종합한 mAP를 핵심 평가지표로 채택함.
- COCO 벤치마크 기준인 mAP50-95를 계산하여 다양한 환경에서도 강건(Robust)하게 결함을 검출하는지 검증함.

### C. 실시간 인라인 검사 처리 능력 평가
- **Inference Latency ($ms/img$)**: 대량 양산 환경에 모델을 적용하기 위해, 이미지 1장당 소요되는 추론 속도(Latency)를 밀리초(ms) 단위로 정밀 측정하여 하드웨어 리소스 대비 효율성을 평가함.

---

## ⚙️ 4. 기술적 방법론 및 실험 설계 (Methodology & Experimental Design)

### A. 비지도 학습 기반 Anomaly Detection 매커니즘
반도체 양산 공정은 수율이 매우 높아 불량 데이터의 절대적 빈도가 낮은 **데이터 불균형(Data Imbalance)** 문제가 발생하며, 신종 불량의 형태를 예측하기 어렵습니다. 본 연구는 이 한계를 극복하기 위해 정상(Normal) 데이터만을 활용해 특징 공간을 학습하는 **비지도 학습(Unsupervised Learning)** 기법을 확립합니다.
1. **임베딩 및 특징 추출 (Deep Learning Feature Extraction)**: 입력된 웨이퍼 영상을 사전 학습된 신경망 백본(`ResNet18`)에 통과시켜 미세 구조 표현 능력이 우수한 고차원 특징 맵(Feature Map)을 추출함.
2. **메모리 뱅크 구축 및 코어셋 샘플링**: 추출된 정상 특징 벡터들을 가상 공간에 매핑하고 Memory Bank를 구축하여 정상 데이터의 분포 범위를 정밀 정의함.
3. **인접 이웃 거리 계산을 통한 결함 판정 (Distance-based Detection)**: 테스트 데이터 입력 시, Memory Bank 내 정상 특징 군집과의 **최단 인접 거리를 계산(Nearest Neighbor Search)**함. 이 거리가 특정 임계값(Threshold)을 초과하여 정상 분포 공간을 벗어날 경우를 결함(Anomaly)으로 판정함.

### B. 알고리즘 최적화를 위한 다각도 비교 실험 계획 (Ablation Study)
현업의 검증 프로세스를 모사하여 다음 4가지 핵심 축을 중심으로 대조 실험을 수행하고 분석 보고서를 작성합니다:

| 실험 축 (Dimension) | 실험 대조군 1 (Baseline) | 실험 대조군 2 (Proposed) | 실험 목적 |
| :--- | :--- | :--- | :--- |
| **전처리 알고리즘** | 원본 영상 데이터 직접 학습 | **CLAHE** 기반 국소 대비 강화 영상 학습 | 전처리가 미세 결함 시각화 및 검출력에 미치는 영향 검증 |
| **가중치 초기화 조건** | 사전 학습된 가중치 활용 (**Transfer Learning**) | 가중치 없이 처음부터 학습 (**Learning from Scratch**) | 전이 학습이 수렴 속도 및 최종 정확도에 미치는 기여도 분석 |
| **모델 아키텍처** | 범용 객체 탐지 모델 (**YOLOv8 / YOLO11**) | 딥러닝 특징 거리 기반 모델 (**PatchCore**) | 상용 탐지 알고리즘과 공정 특화 알고리즘 간의 효율성 비교 |
| **네트워크 하이퍼파라미터** | 얕은 레이어 수 & 고정 커널 크기 | **깊은 레이어 수 & 다중 커널 크기** 변경 | 딥러닝 수용 영역(Receptive Field) 변화가 미세 패턴 추출에 미치는 영향 분석 |

---

## 📋 5. 통합 개발 로드맵 (Checklist & Roadmap)

### 1) 데이터 수집 및 지식 베이스 (Data & Preprocessing)
- [x] **[D-1] 원본 이미지 데이터셋 확보 및 로드 스크립트 구현**
  - **대상**: MVTec AD (Bottle/Capsule 등 주요 오브젝트 카테고리)
  - **기술**: gdown 연동 다운로드 자동화 스크립트 작성 및 zip 해제
- [ ] **[D-2] 공정 특화 영상 전처리 파이프라인 구축**
  - **대상**: 미세 스크래치 및 패턴 오류 영역 강조 이미지
  - **기술**: OpenCV 기반 **CLAHE** 객체 생성 및 타일 그리드 사이즈별 파라미터 변환 적용
- [ ] **[D-3] 벤치마크 정답(Ground-Truth) 마스크 매핑**
  - **대상**: 테스트용 불량 위치 마스크(0: 정상, 255: Defect) 가공
  - **기술**: 수작업으로 라벨링된 실제 불량 영역 영상 매칭 및 전처리 통일

### 2) 딥러닝 모델 설계 및 비지도 학습 구현 (AI/Model)
- [ ] **[M-1] 재구성 기반 이상 탐지 모델 설계 (AutoEncoder)**
  - **기술**: PyTorch 기반 `ConvAutoEncoder` 클래스 구현
  - **핵심**: 정상 이미지 압축 후 복원 프로세스 구축 및 **MSELoss** 기반 Reconstruction Error 로직 정의
- [ ] **[M-2] 딥러닝 Feature 공간 거리 기반 모델 설계 (PatchCore)**
  - **기술**: 사전 학습된 신경망 백본(Pretrained Backbone: ResNet18) 특징 추출 연동
  - **핵심**: Memory Bank 구축 및 최단 인접 이웃 탐색(Nearest Neighbor Search) 알고리즘 연동
- [ ] **[M-3] 백그라운드 학습 자동화 및 로그 적재**
  - **기술**: 원격 터미널 연결 종료 대비 프로세스 관리
  - **핵심**: `nohup python train.py > train.log 2>&1 &` 명령어 실행을 통한 `train.log` 백그라운드 기록 파이프라인 구축

### 3) 성능 정량 평가 및 알고리즘 최적화 (Metrics & Evaluation)
- [ ] **[E-1] 정량적 성능 지표 산출 모듈 구현**
  - **정밀도 검증**: 예측 영역과 정답 영역의 공간적 중첩도를 판정하는 **IoU** 연산 코드 구현
  - **정확도 검증**: 결함 유형별 정밀도와 재현율을 종합한 **mAP** 스코어 산출 시스템 연동
  - **실시간성 검증**: 인라인 자동 검사를 위한 이미지 1장당 추론 속도(**Inference Latency**) 측정 로직 구현
- [ ] **[E-2] 진행 상황 중간 점검 및 교수님 개인 면담 (6월 3일 예정)**
  - **핵심**: 깃허브 커밋 기록과 초기 `train.log` 결과물을 바탕으로 전처리 기법 및 하이퍼파라미터 튜닝 방향성 디벨롭 면담 진행

---

## 🏗️ 6. 시스템 아키텍처 및 코드 구조 (Directory Structure)
```text
├── data/               # WM-811K 및 MVTec AD 반도체 데이터셋 아카이브
├── preprocessing/      # CLAHE, 대비 평활화 등 영상 데이터 고도화 스크립트
├── models/             # Custom CNN architecture 및 PatchCore 알고리즘 정의 파이썬 파일
├── logs/               # train.log 백그라운드 학습 데이터 및 에포크별 정량 지표 저장 폴더
└── README.md           # 연구 목적, 방법론, 실험 결과 기술 보고서
