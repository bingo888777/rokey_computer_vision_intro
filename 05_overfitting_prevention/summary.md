# 과적합 방지 (Overfitting Prevention)

## 1. 과적합 / 과소적합 진단

| 상태 | 증상 |
|---|---|
| 과적합(Overfitting) | Train Loss↓ but Val Loss↑ → 학습-검증 간 Loss 괴리 확대 |
| 과소적합(Underfitting) | Train Loss도 높고 Val Loss도 높음 → 모델 표현력 부족 |
| 이상적 상태 | Train Loss ≈ Val Loss, 둘 다 낮게 수렴 |

## 2. 정규화(Regularization) 기법 비교

| 기법 | 원리 | 효과 | PyTorch |
|---|---|---|---|
| L1 Regularization | 가중치 절댓값 합을 Loss에 추가 | 희소 모델(일부 w=0) | 수동 구현 |
| L2 Regularization | 가중치 제곱합을 Loss에 추가 | 가중치 전반 감소 | `optim`에 `weight_decay` |
| Dropout | 학습 시 뉴런을 랜덤 비활성화 | 앙상블 효과, 공동 적응 방지 | `nn.Dropout(p=0.5)` |
| Batch Normalization | 미니배치 내 정규화 후 학습 | 빠른 학습, 학습률 민감도↓ | `nn.BatchNorm2d` |
| Early Stopping | Val Loss 증가 시 학습 중단 | 최적 시점 모델 저장 | 수동/callback |

> ✅ 표준 레시피: BatchNorm → Dropout 순으로 적용. Dropout은 학습 시만 활성(`model.train()`), 추론 시 비활성(`model.eval()`).

## 3. 그 외 안정화 요소

| 상황 | 대응 |
|---|---|
| 기울기 소실 | ReLU로 해결 / Skip Connection(ResNet)으로 근본 해결 |
| 학습 불안정 | BatchNorm, 학습률 스케줄러, 가중치 초기화(He/Xavier) |
| 클래스 불균형 | 클래스 가중치(weight=...), Focal Loss, 오버샘플링(SMOTE) |

## 4. 핵심 공식

- L2 정규화: `L_total = L + λ·‖W‖₂²`
- Dropout: 학습 시 각 뉴런을 확률 p로 비활성 / 추론 시 전체 뉴런 사용, 출력에 (1-p) 곱함

---
*출처: 컴퓨터비전 개론 12, 17차시 재구성*
