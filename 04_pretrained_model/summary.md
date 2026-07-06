# 사전학습 모델 (Pretrained Model) & 전이학습

## 1. 주요 사전학습 모델 발전사

### 1.1 AlexNet (2012)
| 항목 | 내용 |
|---|---|
| 의의 | ImageNet LSVRC 2012 우승, 딥러닝 르네상스 시작 |
| 구조 | 5개 Conv층 + 3개 FC층, 약 6000만 파라미터 |
| 혁신 1 | ReLU 활성화 함수 → Sigmoid 대비 6배 빠른 학습 |
| 혁신 2 | Dropout(p=0.5) 정규화 도입 |
| 혁신 3 | LRN(Local Response Normalization) 적용 |
| 혁신 4 | GPU 병렬 학습(GTX 580×2)으로 대규모 학습 실현 |

### 1.2 GoogLeNet / Inception v1 (2014)
| 항목 | 내용 |
|---|---|
| 의의 | ImageNet LSVRC 2014 우승, Top-5 Error 6.7% |
| 핵심 아이디어 | Inception Module: 1×1, 3×3, 5×5 필터 병렬 적용 후 채널 방향 Concatenate |
| 1×1 Conv | 채널 수 축소(Bottleneck) → 연산량 대폭 감소 |
| 보조 분류기 | 중간 층에도 Classifier 추가 → 기울기 소실 완화 |
| 파라미터 | 약 500만 (AlexNet 대비 1/12로 경량화) |

### 1.3 VGGNet (2014)
| 항목 | 내용 |
|---|---|
| 의의 | 단순 반복 구조로 깊이의 중요성 증명(16~19층) |
| 핵심 원칙 | 3×3 Conv만 사용, 필터 크기 통일 |
| 3×3의 이유 | 3×3 두 번 쌓으면 5×5와 동일한 수용 영역이나 파라미터는 더 적음 |
| 한계 | 1억 4천만 파라미터로 메모리 소비 큼 |
| 활용 | Transfer Learning의 feature extractor로 널리 활용 |

### 1.4 ResNet (2015)
| 항목 | 내용 |
|---|---|
| 의의 | ILSVRC 2015 우승, 152층 딥 네트워크 학습 성공 |
| 핵심 아이디어 | 잔차 연결(Residual/Skip Connection): F(x) + x |
| 문제 해결 | 기울기 소실 → 지름길로 기울기 직접 전달, 깊은 층 학습 가능 |
| Bottleneck | 1×1→3×3→1×1 구조로 파라미터 절감(ResNet-50 이상) |
| 성능 | Top-5 Error 3.57% (인간 수준 5%보다 낮음) |

### 1.5 모델 비교 요약
| 모델 | 연도 | 깊이 | 파라미터 | Top-5 Error |
|---|---|---|---|---|
| AlexNet | 2012 | 8층 | ~60M | 15.3% |
| VGG-16 | 2014 | 16층 | ~138M | 7.3% |
| GoogLeNet | 2014 | 22층 | ~5M | 6.7% |
| ResNet-50 | 2015 | 50층 | ~25M | 5.25% |
| ResNet-152 | 2015 | 152층 | ~60M | 3.57% |

## 2. 전이학습(Transfer Learning) 전략

- 사전학습 모델 가중치를 초기값으로 활용
- ImageNet의 일반 특징(엣지, 질감, 형태)이 다른 도메인에도 유용

| 전략 | 방법 | 적합 상황 |
|---|---|---|
| Feature Extraction | Backbone 가중치 동결(freeze), 분류기만 학습 | 소규모 데이터셋 |
| Fine-tuning | 전체 또는 일부 층 미세 조정(낮은 lr) | 중간 규모 데이터셋 |
| Scratch 학습 | 처음부터 전체 학습 | 대규모 데이터셋 + 도메인 특수 |

### 심화 Fine-tuning 전략
| 전략 | 설정 | 데이터 규모 | 방식 |
|---|---|---|---|
| Partial Fine-tune | 상위 N개 블록만 unfreeze | 중간(1K~10K) | lr=1e-4로 상위 층 미세 조정 |
| Full Fine-tune | 전체 파라미터 학습 | 충분(>10K) | lr=1e-5~1e-4, 점진적 해동 |

## 3. Custom Dataset 구현

- `torch.utils.data.Dataset` 상속
- `__len__()`: 데이터셋 전체 크기 반환
- `__getitem__(idx)`: 인덱스에 해당하는 (이미지, 레이블) 반환
- DataLoader에 커스텀 Dataset 전달, `batch_size`·`shuffle` 설정

## 4. 데이터 증강 (Advanced)

| 기법 | 설명 | 효과 |
|---|---|---|
| RandomCrop / CenterCrop | 이미지 랜덤/중앙 자르기 | 위치 불변성 학습 |
| ColorJitter | 밝기·대비·채도·색조 랜덤 변환 | 조명 변화 강건성 |
| RandomHorizontalFlip | 좌우 반전(50% 확률) | 좌우 대칭 데이터 확보 |
| Mixup | 두 샘플을 α비율로 선형 혼합 | 결정 경계 부드럽게 |
| CutMix | 한 이미지 영역을 다른 이미지로 교체 | 배경 의존성 감소 |
| AutoAugment | 강화학습으로 최적 증강 정책 탐색 | 수작업 튜닝 대체 |

> ✅ Mixup/CutMix는 레이블도 혼합: `y = λ·y_A + (1−λ)·y_B` (Soft Label로 CrossEntropy 계산)

---
*출처: 컴퓨터비전 개론 13, 14, 15차시 / CV응용 심화 05차시 재구성*
