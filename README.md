# ✍️ Auto CNN Re-Training MLOps with Docker

사용자 데이터를 기반으로 CNN 모델을 자동 재학습하고, 최신 모델을 배포하는 **MLOps 파이프라인 프로젝트**입니다.  

### [📑 PPT](https://drive.google.com/file/d/1PhzmUYa8kG9stoeywFc_wLl2aUckNSYq/view?usp=drive_link) | [🎥 Docker를 활용한 MLOps 구축 시연](https://youtu.be/POb1gV_6WkM)

> ⚠️ 본 streamlit 사이트는 ngrok으로 로컬을 외부에 공개한 것이라 보안상 프로젝트 진행 시에만 열어두며, 평상시에는 닫혀 있으니 시연은 영상을 참조해 주세요.

---


## 📊 Demo & Dashboard

[<img src="https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" width="180"/>](https://fdc715794a4f.ngrok-free.app/)

![Demo Dashboard](images/dashboard.png)

#### Metrics 설명
- **Clean**  
  : 변형을 주지 않은 원본 데이터(깨끗한 입력)로 평가한 성능  
- **Robust**  
  : 회전(±15°), 노이즈(0.15) 등 **증강 데이터(변형된 입력)** 를 사용해 평가한 성능
  
---

## 🏗 Architecture

<p align="center">
  <img src="images/architecture.png" width="600"/>
</p>

- **Modeling**: CNN 학습 및 재학습 (MNIST + 사용자 데이터)  
- **Trigger**: 일정 시간/조건에 맞춰 자동 재학습 실행  
- **Streamlit**: 성능 모니터링 및 시각화 제공  
- **Share Storage**: 로그·데이터·모델 버전 관리  

---


## 🧠 Model
본 프로젝트의 기본 모델은 초기 학습에서 CNN 기반의 MNIST 손글씨 숫자 분류기를 사용하였습니다.  

- **초기 학습 (Pre-training)**  
  - MNIST 학습 데이터 60,000장 활용  
  - Conv2D → ReLU → MaxPooling → Flatten → Dense → Softmax 구조  

<p align="center">
  <img src="images/cnn_structure.png" width="500"/>
</p>


- **자동 재학습 (Re-training)**  
  - 사용자가 업로드한 피드백 데이터 + 기존 데이터 일부(리허설 데이터)를 함께 학습  
  - Trigger 컨테이너가 주기적으로 실행되어 새로운 데이터를 반영  
  - 학습된 모델은 `share_storage`에 버전별(`v1`, `v2`, …)로 저장 및 추적 가능  


---


## 📦 Dataset
- **초기 학습**: MNIST 학습 60,000장 / 테스트 10,000장  
- **재학습**: 사용자 피드백 데이터 + MNIST 리허설 데이터  
- **평가 방식**  
  - Clean Test: 원본 MNIST  
  - Aug Test: 회전(±15°) + 노이즈 추가  


---

## 📂 Repository Structure
``` bash
TECH_SEMINAR_DM/
├── modeling/         # CNN 모델 학습 및 재학습 코드
├── share_storage/    # 모델·로그·데이터 공유 스토리지
├── streamlit_app/    # 성능 모니터링 및 시각화 대시보드
├── trigger/          # 주기적/조건부 재학습 트리거 실행
├── .dockerignore     # Docker 빌드 시 제외 파일 정의
└── docker-compose.yml # 전체 파이프라인 서비스 관리
```

---


## ✨ Features
- 🖼 **이미지 업로드** → CNN 모델이 예측 (결과 + 확률)  
- 📝 **라벨 입력** → 사용자가 정답 제공 (피드백 데이터 저장)  
- 🔄 **자동 재학습** → Trigger 컨테이너가 새로운 데이터 감지 후 CNN 재학습  
- 📈 **실시간 모니터링** → Streamlit 대시보드에서 accuracy/loss 추적  
- 🗂 **버전 관리** → 모델 학습 기록 및 이전 버전 복원 가능  


---


## 📚 Concept
- **MLOps 적용**: 학습·재학습·배포·모니터링 자동화  
- **Feedback 기반 학습**: 사용자 입력 데이터를 반영해 모델 개선  
- **Docker 기반 분리**: 학습/배포/대시보드 환경 독립 실행  


---


### 📦 Share Storage
- **history**: 학습 버전별 accuracy/loss 기록  
- **uploads**: 사용자 입력 데이터 저장  
- **data/backupdata**: 재학습 데이터와 백업 데이터 분리  
<p align="center">
  <img src="images/logging.png" width="850" height="350"/>
</p>

---


### ⏰ Trigger
- 매일 **11:50** 자동 실행 (신규 데이터 없으면 skip)  
- 재학습 후 성능 기록 & 이전 모델과 비교 가능  
- 버전: v1, v2, v3 … 추적 및 복원 가능
<p align="center">
  <img src="images/modeling_retraining_workflow.png" width="656" height="567"/>
</p>



---


### 🐳 Docker Compose
| Service      | Role              | 특징                       |
|--------------|-------------------|----------------------------|
| **streamlit** | 결과 시각화 및 예측 제공 | Web UI (포트 8501)        |
| **trigger**   | 재학습 트리거 실행       | 실행 시간·조건 관리       |
| **modeling**  | CNN 학습 & 테스트       | MNIST + 사용자 데이터 학습 |

---


## 🐞 Issues & Solutions
| Issue | Problem | Solution |
|-------|---------|----------|
| 데이터 불균형 | 신규 사용자 데이터만 쓰면 과적합 위험 | 기존 MNIST 데이터 일부 + 사용자 데이터 혼합 재학습 |
| 데이터 누수 | Feedback 데이터가 검증셋에 포함될 위험 | Feedback 데이터는 테스트셋에서 완전 제외 |
| 모델 관리 어려움 | 최신 모델만 유지하면 추적 불가 | `history` + `backup-data` 구조로 버전 관리 |
| 일관성 없는 실행 환경 | 로컬 환경 차이 | Docker Compose 기반 컨테이너화 |


---


## 🔮 Future Work
- 📊 **로그/모니터링 고도화**: 성능 이상 빠른 탐지  
- ☸ **배포 체계 강화**: Kubernetes 확장 적용  
- 🏆 **최적 모델 자동 선택**: 가장 성능 좋은 모델 자동 배포  


---


## 👨‍👩‍👧‍👦 Team & Roles

| 이름   | 역할 |
|--------|------------------------------------------------|
| 김은수 | 모델 배포 및 파이프라인 조율 |
| 김유라 | CNN 모델링 및 자동 재학습 구현 |
| 김지윤 | Trigger 설계 및 자동화 파이프라인 구축 |
