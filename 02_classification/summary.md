# 분류 머신러닝 (Classification)

## 1. 활성화 함수 비교

| 함수 | 수식 | 특징 | 주 사용처 |
|---|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | 출력 (0,1), 기울기 소실 문제 | 이진 분류 출력층 |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 출력 (-1,1), Sigmoid보다 빠름 | 은닉층(레거시) |
| ReLU | max(0, x) | 빠른 학습, Dead Neuron 문제 | 은닉층(표준) |
| LeakyReLU | max(0.01x, x) | Dead Neuron 완화 | 은닉층(개선형) |
| Softmax | eˣᵢ/Σeˣⱼ | 합=1, 확률로 해석 | 다중 분류 출력층 |

> ⚠️ 기울기 소실(Vanishing Gradient): Sigmoid를 깊게 쌓으면 앞쪽 층 기울기 ≈ 0 → ReLU로 해결

## 2. 이진 분류 (Binary Classification)

### 2.1 손실함수
| 손실함수 | 수식 | PyTorch API |
|---|---|---|
| BCE | −[y·log(ŷ)+(1−y)·log(1−ŷ)] | `nn.BCELoss()` |
| BCEWithLogits | BCE + Sigmoid 통합(수치 안정적) | `nn.BCEWithLogitsLoss()` |

### 2.2 혼동행렬(Confusion Matrix)
| | 예측 Positive | 예측 Negative |
|---|---|---|
| 실제 Positive | TP | FN |
| 실제 Negative | FP | TN |

### 2.3 평가 지표
| 지표 | 수식 | 의미 | 중요 케이스 |
|---|---|---|---|
| Accuracy | (TP+TN)/전체 | 전체 정확도 | 클래스 균형 데이터 |
| Precision | TP/(TP+FP) | 양성 예측의 정확도 | FP를 줄여야 할 때(스팸필터) |
| Recall | TP/(TP+FN) | 실제 양성 탐지율 | FN을 줄여야 할 때(암 진단) |
| F1-Score | 2·P·R/(P+R) | Precision·Recall 조화평균 | 불균형 데이터 |
| ROC-AUC | TPR vs FPR 곡선 면적 | 임계값 무관 성능 | 모델 비교 |

> ⚠️ 클래스 불균형 시 Accuracy는 의미가 없음. Precision/Recall/F1 또는 ROC-AUC 사용 권장.

## 3. 다중 분류 (Multi-class Classification)

### 3.1 Softmax와 Cross-Entropy
- Softmax: 출력값을 합=1인 확률 분포로 변환
- Cross-Entropy Loss: 예측 확률 분포와 실제 레이블 차이를 측정 → L = −log(p_정답클래스)

| 손실함수 | 수식 | PyTorch API |
|---|---|---|
| CrossEntropy | −Σ yᵢ·log(ŷᵢ) | `nn.CrossEntropyLoss()` |
| NLLLoss | −log(ŷ_class) | `nn.NLLLoss()` (LogSoftmax 후 사용) |

| 기법 | 설명 |
|---|---|
| 원-핫 인코딩 | 클래스 k → [0,0,...,1,...,0] 벡터 표현 |
| Label Smoothing | 정답 레이블 y=1 대신 y=0.9로 설정 → 과신 방지, 일반화 향상 |
| Temperature Scaling | Softmax 적용 전 logits/T로 나눠 확률 분포 조절 |

### 3.2 대표 데이터셋 (MNIST / FashionMNIST)
| 항목 | MNIST | FashionMNIST |
|---|---|---|
| 클래스 수 | 10 (숫자 0-9) | 10 (의류 카테고리) |
| 이미지 크기 | 28×28 흑백 | 28×28 흑백 |
| 샘플 수 | 학습 60K / 테스트 10K | 학습 60K / 테스트 10K |
| 난이도 | 낮음 | 중간 |

### 3.3 표준 학습 파이프라인
1. DataLoader: Dataset → mini-batch 자동 생성, `shuffle=True`
2. 전처리: `transforms.ToTensor()`, `transforms.Normalize((0.5,), (0.5,))`
3. MLP 모델: Flatten → Linear → ReLU → Linear → Softmax
4. 학습 루프: for epoch / for batch → forward → loss → backward → step
5. 검증: `model.eval()` + `torch.no_grad()` → 기울기 계산 비활성화

---
*출처: 컴퓨터비전 개론(딥러닝 입문) 총정리 04, 06, 08, 09, 10차시 재구성*
