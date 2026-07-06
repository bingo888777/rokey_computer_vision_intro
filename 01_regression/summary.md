# 회귀 머신러닝 (Regression)

## 0. 사전 지식 (딥러닝 실습 공통 기초)

### 0.1 OOP 기초
PyTorch는 객체지향(OOP) 구조로 설계되어 있어, 아래 개념을 알아야 코드를 읽고 쓸 수 있다.

| 개념 | 설명 | 딥러닝 활용 예시 |
|---|---|---|
| 클래스(Class) | 객체의 설계도 | `nn.Module` 상속으로 모델 정의 |
| 인스턴스(Instance) | 클래스로부터 생성된 실체 | `model = MyNet()` |
| 상속(Inheritance) | 부모 클래스 기능 재사용 | `class MyNet(nn.Module)` |
| 메서드 오버라이딩 | 부모 메서드 재정의 | `forward()` 구현 |
| 캡슐화 | 데이터+메서드를 하나로 묶음 | `self.layers` 내부에 감춤 |

### 0.2 텐서(Tensor)와 자동미분(Autograd)
- 텐서 차원: 0차원(스칼라) → 1차원(벡터) → 2차원(행렬) → 3차원+(이미지 C,H,W / 배치 N,C,H,W)
- `requires_grad=True` 설정 시 모든 연산이 추적되고, `.backward()` 호출로 기울기 자동 계산
- 순전파(계산 그래프 구축) → 역전파(체인룰로 기울기 계산)
- `optimizer.zero_grad()`는 매 스텝 기울기 초기화를 위해 필수

### 0.3 신경망 기초 (퍼셉트론 → MLP)
| 항목 | 설명 |
|---|---|
| 퍼셉트론 | 입력×가중치의 합 → 임계값 비교 (선형 분류기) |
| 다층 퍼셉트론(MLP) | 은닉층 추가 → 비선형 문제 해결 가능 |
| 역전파 | 체인룰 적용(∂L/∂w = ∂L/∂z · ∂z/∂w), 출력층→입력층 방향으로 기울기 전파 |

---

## 1. 선형 회귀 (Linear Regression)

| 항목 | 내용 |
|---|---|
| 모델 수식 | ŷ = Xw + b |
| 목적 | MSE(평균 제곱 오차) 최소화 |
| OLS 해석해 | w* = (XᵀX)⁻¹Xᵀy (정규방정식) |
| 평가 지표 | MSE, RMSE, MAE, R²(결정계수) |

### 학습 절차
1. 데이터 정규화(Standardization): `(x - μ) / σ`
2. 모델 정의: `nn.Linear(in_features, out_features)`
3. 손실함수: `nn.MSELoss()`
4. 옵티마이저: `optim.SGD` 또는 `optim.Adam`
5. 에포크 반복: forward → loss → backward → step

> ✅ 특성 스케일이 다르면 학습이 불안정해질 수 있으므로 항상 StandardScaler 또는 Min-Max Scaling 적용.

## 2. 손실함수 (회귀 관련)

| 손실함수 | 수식 | PyTorch API |
|---|---|---|
| MSE | 평균((ŷ−y)²) | `nn.MSELoss()` |

## 3. 최적화 알고리즘 (경사하강법)

| 알고리즘 | 핵심 | 장점 | 단점 |
|---|---|---|---|
| Batch GD | W ← W − η·∇L | 안정적 수렴 | 대용량 데이터 비효율 |
| SGD | 랜덤 1개 샘플로 업데이트 | 빠른 업데이트 | 노이즈 많음 |
| Mini-batch GD | 랜덤 B개 샘플 | GPU 효율 + 안정성 균형 | 배치 크기 튜닝 필요 |
| Momentum | v = γv − η∇L | 진동 감소 | 하이퍼파라미터 추가 |
| Adam | Momentum+RMSProp 결합 | 빠르고 안정적, 기본 추천 | 메모리 소비 큼 |

### 학습률(Learning Rate)
- 너무 크면: Loss 발산 (overshooting)
- 너무 작으면: 학습 느림, Local Minimum에 갇힘
- 스케줄러: StepLR, CosineAnnealingLR, ReduceLROnPlateau

> ✅ 실전 기본값: Adam + lr=1e-3, 성능 미달 시 lr=1e-4로 조정

## 참고 실습
- Boston Housing 데이터셋 기반 회귀 실습

---
*출처: 컴퓨터비전 개론(딥러닝 입문) 총정리 07차시, 05차시, 06차시 재구성*
