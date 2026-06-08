# 🔎 딥러닝 기반 반도체 웨이퍼 결함 탐지 및 유형 분류 연구
> **Semiconductor Wafer Defect Detection and Classification using Deep Learning Features**

## 🎯 1. 프로젝트 개요 및 연구 동기 (Introduction)
- **주제**: 이미지 비전 센서 데이터를 기반으로 반도체 공정 중 발생하는 웨이퍼 결함을 실시간 탐지(Detection)하고 유형을 분류(Classification)함.
- **연구 동기**: `2025 국제인공지능산업대전(AI EXPO KOREA)`, `2026 국제인공지능산업대전(AI EXPO KOREA)` 참관을 통해 대량 양산 및 미세 공정 최적화 흐름 속에서 AI 비전 검사 솔루션이 반도체 수율(Yield) 향상의 핵심 동력임을 직접 목격함. 이에 최신 컴퓨터 비전 방법론을 반도체 도메인에 적용하여 실무적 난제를 해결하고자 본 연구를 기획하고 성공적으로 완수함.

---

## 🏗️ 2. 프로젝트 아키텍처 및 연구 환경 (System Architecture)
본 연구는 대규모 반도체 웨이퍼 데이터셋의 안정적인 학습 및 고속 추론 성능을 보장하기 위해, 고성능 클라우드 인프라 기반의 End-to-End 딥러닝 파이프라인을 가동하였습니다.

- **컴퓨팅 인프라 (Environment)**: AWS EC2 (`g4dn.xlarge`) 하이브리드 인스턴스 (Ubuntu 24.04 LTS)
- **핵심 가속 장치 (GPU)**: NVIDIA T4 Tensor Core GPU (16GB VRAM)
- **선행 연구 환경 (Prototyping)**: Google Colab GPU 환경을 통한 초기 EDA 및 데이터 파이프라인 가공 선행 완료
- **스토리지 (Storage)**: AWS EBS (Elastic Block Store) 고속 볼륨 연동
- **가상 환경 (Sandbox)**: Miniconda 독립 가상환경 (`anomaly_env`) 구축을 통한 의존성 충돌 방지
- **딥러닝 프레임워크**: PyTorch (`torch`, `torchvision`) 기반 종단간 모델링
- **데이터 파이프라인 (Data)**: 반도체 웨이퍼 맵 대용량 데이터셋 (`WM-811K`) 및 전처리 `SemiconductorDataset` 스트리밍 파이프라인 구현
- **프로세스 관리 및 로깅 (M&E)**: 리눅스 백그라운드 프로세스 기반 구동 (`nohup`) 및 실시간 `nohup.out` 아카이빙 로직 구현

---

## 📅 3. 프로젝트 일정 및 진행 계획 (Checklist & Roadmap)
교수님 피드백 및 학사 일정에 맞춘 반도체 웨이퍼 결함 탐지 프로젝트의 최종 발표 로드맵입니다.

### 📌 주요 일정 및 마일스톤
- [x] **05.25~26** : **Google Colab 기반 선행 프로토타입 연구 및 데이터셋 무결성 정제**
  - Google Colab 환경 연동을 통해 WM-811K 반도체 대규모 데이터셋 EDA 선행 완료
  - 기본 데이터 가공 및 파이토치 기반 `Pytorch_cv_project.ipynb` 초기 파이프라인 검증 완수
  - 11.5만 장 대규모 정상 데이터셋 정규화 및 가공 로직 설계 완료
- [x] **06.02** : **AWS 가상 머신 이주 및 교수님 대면 중간 점검 피드백** 🔍
  - *수행 항목*: AWS EC2 고성능 GPU 인프라 환경 구축 완료 및 AutoEncoder 초기 모델 스코어 중간 점검 완수
  - **👍 교수님 주요 피드백 및 수정 요구사항**:
    - **[데이터 효율화]** 대규모 14,000장 테스트셋은 불필요함. 대신 정상 비율(84%)이 높은 공정 특성을 반영하되, 불량 유형이 정밀하게 포함된 압축 시험지 구성 필요
    - **[모델 확장성]** 단일 오토인코더 구조에 국한되지 말고, **PatchCore(교수님 강력 강조)** 및 Vision Transformer(ViT) 등 최신 비지도 학습 알고리즘 패러다임을 연동하여 다각도로 교차 비교할 것
    - **[클라우드 고도화]** 대용량 연산의 이점을 살려 **AWS 인프라 및 가속기 환경**을 최대한 적극적으로 활용하여 벤치마크할 것
- [x] **06.09 (현재 주간)** : **보강 주간 및 피드백 기반 3종 경기 모델 총공세 (수업 없음)** 💻
  - **피드백 100% 반영 완료**: 정상/불량이 정교하게 혼합된 **최종 파이널 시험지 3,600장** 구축 완료
  - **AWS 인프라 성능 극대화**: AWS EC2 `g4dn.xlarge` 및 T4 GPU 가속기를 기반으로 `nohup` 백그라운드 멀티 프로세스 가동
  - **3대 알고리즘 교차 벤치마크 달성**: 
    - Baseline AutoEncoder 가동 및 한계 분석 완료 (`49.40%`)
    - **[Proposed] PatchCore 메모리 뱅크 엔진 구현 및 불량 완벽 탐지 완료 (`100.00%`)**
    - Vision Transformer(ViT) Attention 기반 이상 탐지 도출 완료 (`98.72%`)
- [ ] **06.16** : **최종 프로젝트 발표 및 포트폴리오 배포** 🚀
  - *주요 목표*: 최종 3종 모델 대조 벤치마크 결과 시연, GitHub 원격 저장소 최종 정돈 완료
  - *제출 자료 마감*: 최종 기말 발표자료, 프로젝트 보고서 PDF, 모델별 최종 학습 및 추론 `result_summary.txt` 로그 제출

---

## 📊 4. 성능 정량 평가 지표 및 기준 (Quantitative Evaluation Metrics)
현업 인라인(In-line) 자동 검사 시스템(ADC) 도입 가능성을 타당성 있게 검증하기 위해 학술적·실무적 관점에서 다음과 같은 명확한 평가 지표를 적용하여 최종 모델들을 검증했습니다.

### A. 픽셀 단위 결함 국소화(Localization) 능력 평가
- **IoU (Intersection Over Union)**: 가상 특징 공간 내 거리 분석을 통해 모델이 예측한 결함 영역(Predicted)과 실제 결함의 정답 영역(Ground-Truth)의 공간적 중첩도를 측정함.
- 本 연구에서는 $IoU \ge 0.5$ 이상을 유효 검출(True Positive)의 최소 기준으로 정의하여, 현업의 치명적 이슈인 오탐지율(False Positive Rate)을 최소화하는 최적의 임계값(Threshold)을 도출함.

### B. 다중 결함 검출 및 분류 성능 평가
- **mAP (Mean Average Precision) @50 및 @50:95**: 반도체 공정 특성상 불량의 클래스가 다양하므로(Scratch, Center, Edge-Ring, Loc 등), 각 결함 유형별 정밀도(Precision)와 재현율(Recall)을 종합한 mAP를 핵심 평가지표로 채택함. COCO 벤치마크 기준인 mAP50-95를 기반으로 삼아 다양한 환경에서도 강건(Robust)하게 결함을 검출하는지 검증함.

### C. 실시간 인라인 검사 처리 능력 평가
- **Inference Latency ($ms/img$)**: 대량 양산 환경에 모델을 적용하기 위해, 이미지 1장당 소요되는 추론 속도(Latency)를 밀리초(ms) 단위로 정밀 측정하여 하드웨어 리소스 대비 효율성을 평가함.

---

## ⚙️ 5. 기술적 방법론 및 실험 설계 (Methodology & Experimental Design)

### A. 비지도 학습 기반 Anomaly Detection 매커니즘
반도체 양산 공정은 수율이 매우 높아 불량 데이터의 절대적 빈도가 낮은 **데이터 불균형(Data Imbalance)** 문제가 발생하며, 신종 불량의 형태를 예측하기 어렵습니다. 본 연구는 이 한계를 극복하기 위해 정상(Normal) 데이터만을 활용해 특징 공간을 학습하는 **비지도 학습(Unsupervised Learning)** 기법을 확립하였습니다.

1. **임베딩 및 특징 추출 (Deep Learning Feature Extraction)**: 입력된 웨이퍼 영상을 사전 학습된 신경망 백본(`ResNet18`)에 통과시켜 미세 구조 표현 능력이 우수한 고차원 특징 맵(Feature Map)을 추출함.
2. **메모리 뱅크 구축 및 코어셋 샘플링**: 추출된 정상 특징 벡터들을 가상 공간에 매핑하고 Memory Bank를 구축하여 정상 데이터의 분포 범위를 정밀 정의함.
3. **인접 이웃 거리 계산을 통한 결함 판정 (Distance-based Detection)**: 테스트 데이터 입력 시, Memory Bank 내 정상 특징 군집과의 **최단 인접 거리를 계산(Nearest Neighbor Search)**함. 이 거리가 특정 임계값(Threshold)을 초과하여 정상 분포 공간을 벗어날 경우를 결함(Anomaly)으로 판정함.

### B. 알고리즘 최적화를 위한 다각도 비교 실험 및 최종 결과 (Ablation Study)
교수님 피드백을 반영하여 **알고리즘 패러다임 자체를 전환하는 3대 비전 모델 교차 벤치마크 실험**을 완수하였습니다. 

동일한 대규모 학습 데이터셋(**정상 115,544장**)으로 각 엔진을 훈련한 후, 9종의 결함 유형과 정상 데이터가 고스란히 혼합된 **실전 파이널 시험지 3,600장**을 대상으로 최종 분류 정확도($\text{Accuracy}$)를 도출한 결과는 다음과 같습니다.

| 실험 비교 축 (Dimension) | 적용 알고리즘 및 구조적 특징 | 가중치 초기화 조건 | 테스트 데이터셋 (Test) | 최종 분류 정확도 ($\text{Accuracy}$) | 연구 결론 및 핵심 분석 (Ablation Summary) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **실험 대조군 1 (Baseline)** | **AutoEncoder** (이미지 압축 후 복원 기반 MSE 손실 측정 구조) | From Scratch (처음부터 학습) | 3,600 samples | **49.40%** | 정상 웨이퍼의 미세 노이즈나 무늬 변화에도 과민반응하여 불량으로 오분류하는 한계 노출. |
| **실험 대조군 2** | **Vision Transformer (ViT)** (글로벌 맥락 패치 간 Attention 파악 체계) | Transfer Learning (사전학습 가중치) | 3,600 samples | **98.72%** | 거시적 결함 탐지율은 매우 우수하나, 반도체 특유의 미세한 선형 결함(Scratch)이 맥락에 묻혀 미세 오판 발생. |
| **실험 대조군 3 (Proposed)** | **PatchCore** (국소 패치 임베딩 및 메모리 뱅크 이웃 거리 대조 구조) | Transfer Learning (사전학습 가중치) | 3,600 samples | **100.00%** | 정상 분포 공간과의 기하학적 거리를 칼같이 계산함으로써 미세 불량까지 완벽히 차단, **반도체 도메인에 가장 최적화된 최종 하이엔드 알고리즘으로 검증됨.** |

---

## 📂 6. 시스템 아키텍처 및 코드 구조 (Directory Structure)
프로젝트 인프라 디렉토리를 다음과 같이 정석 구조로 정돈 및 구조화하였습니다.

```text
├── anomaly/
│   ├── Pytorch_cv_project.ipynb # Google Colab 선행 연구 및 초기 프로토타이핑 검증 주피터 노트북
│   ├── anomaly_autoencoder.py   # CNN 기반 이미지 재구성 이상 탐지 가동 스크립트
│   ├── anomaly_patchcore.py     # 고정밀 메모리 뱅크 거리 기반 이상 탐지 메인 엔진
│   ├── anomaly_vit.py           # Multi-Head Attention 기반 트랜스포머 가동 스크립트
│   ├── docs/                    # 기말 최종 발표 자료(PPT) 및 프로젝트 보고서 PDF 아카이브
│   ├── src/                     # 전처리 완료된 반도체 바이너리 데이터셋(train/test_data.pkl) 탑재 폴더
│   ├── colab/                   # colab을 통한 사전 프로젝트 데이터 검증 및 모델 설정 폴더
│   ├── results_autoencoder/     # 각 모델별 실시간 벤치마크 최종 성적표(result_summary.txt) 저장 폴더
│   ├── results_patchcore/
│   └── results_vit/
