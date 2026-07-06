# CNN (Convolutional Neural Network)

## 1. CNN 핵심 연산

| 연산 | 설명 |
|---|---|
| 합성곱(Conv) | 필터(커널)가 이미지를 슬라이딩하며 특징 맵 생성 |
| 출력 크기 공식 | (W−F+2P)/S+1 (W:입력, F:필터, P:패딩, S:스트라이드) |
| 풀링(Pooling) | MaxPool(최댓값 추출) / AvgPool(평균값 추출) → 공간 축소 |
| 특징 추출 | 초기 층: 엣지/색상 → 중간 층: 눈/코 등 부분 패턴 → 깊은 층: 전체 객체 |
| 파라미터 공유 | 동일 필터가 전체 이미지에 적용 → 위치 불변 특성 학습 |

## 2. CIFAR-10 기본 CNN 구조 예시

| 레이어 | 구성 | 출력 크기 |
|---|---|---|
| Conv1 | 3→32 채널, 3×3 커널, ReLU | 32×32×32 |
| MaxPool1 | 2×2 풀링 | 16×16×32 |
| Conv2 | 32→64 채널, 3×3 커널, ReLU | 16×16×64 |
| MaxPool2 | 2×2 풀링 | 8×8×64 |
| Flatten | → 1D 벡터 | 4096 |
| FC+Softmax | 4096→10 | 10 (클래스 확률) |

> CIFAR-10: 32×32 컬러 이미지, 10개 클래스, 학습 50K/테스트 10K

## 3. Residual Block & ResNet-18 스크래치 구현

| 항목 | 설명 |
|---|---|
| 기본 구조 | Conv(3×3)→BN→ReLU→Conv(3×3)→BN→ +x → ReLU |
| Skip Connection | 입력 x를 출력에 더함: out = F(x) + x |
| 크기 불일치 처리 | 채널 수가 다르면 1×1 Conv로 shortcut 투영 |
| 그래디언트 | 덧셈 연산에서 기울기=1 → 기울기 소실 없이 깊은 층까지 전달 |

### ResNet-18 구조 (입력 224×224 기준)
| 블록 | 구성 | 출력 크기 |
|---|---|---|
| Conv1 | 7×7, 64채널, stride=2, MaxPool | 56×56×64 |
| Layer1 | BasicBlock×2 (64→64) | 56×56×64 |
| Layer2 | BasicBlock×2 (64→128, stride=2) | 28×28×128 |
| Layer3 | BasicBlock×2 (128→256, stride=2) | 14×14×256 |
| Layer4 | BasicBlock×2 (256→512, stride=2) | 7×7×512 |
| Head | AvgPool→Flatten→FC(1000) | 1000 |

> ⚠️ 스크래치 구현 시 `nn.Sequential`, `nn.ModuleList` 활용. BatchNorm은 각 Conv 후 필수. 마지막 풀링은 Adaptive Global Average Pooling 권장.

## 4. 설명 가능한 AI (XAI) — CNN 해석

CNN이 왜 그런 결과를 냈는지 시각적으로 설명하는 기법들 (의료·금융 등 고위험 도메인에서 중요).

| 기법 | 원리 | 특징 |
|---|---|---|
| Saliency Map | ∂Loss/∂입력 픽셀 | 픽셀별 민감도 시각화 (Vanilla Gradient는 노이즈 多) |
| SmoothGrad | 가우시안 노이즈 추가 N회 평균 | 더 부드러운 맵 |
| CAM | GAP+FC 가중치로 활성 지도 생성 | 구조 변경(GAP+FC) 필요 |
| Grad-CAM | 마지막 Conv층의 클래스 기울기 GAP → 가중치 계산 | 구조 무관, 어떤 CNN에도 적용 가능 |
| Grad-CAM++ | 픽셀별 기울기 중요도 가중 | 여러 객체 존재 시 더 정확 |
| IG(Integrated Gradients) | 기준점→입력 경로의 적분 기울기 | 완전성 보장, 기준점 선택에 민감 |

> 📝 Grad-CAM 수식: `α_k = (1/Z)·Σ[∂y/∂A_k]`, `L_Grad-CAM = ReLU(Σ α_k · A_k)`

### 의료 영상에서의 활용 (U-Net 기반 분할 + Grad-CAM)
- U-Net: Encoder(수축 경로)+Decoder(확장 경로)+Skip Connection 구조. 소량 데이터로도 우수한 분할 성능
- 흉부 X-ray 폐렴 진단, MRI 뇌종양 분류 시 판단 근거 시각화에 활용
- ⚠️ 의료 영상에서는 False Negative(질병을 놓침)가 치명적 → Recall을 최우선 지표로 사용

## 5. CNN vs Vision Transformer(ViT) 비교

| 항목 | CNN | ViT |
|---|---|---|
| 귀납적 편향 | 지역성, 평행이동 불변성 내재 | 없음(데이터로 학습) |
| 데이터 효율 | 소량 데이터 가능 | 대규모 사전학습 필요 |
| 글로벌 맥락 | 깊게 쌓아야 가능 | 1층부터 전체 포착 |
| 속도 | 빠름 | 대형 모델은 느릴 수 있음 |

> ViT는 이미지를 패치(예: 16×16)로 나눠 토큰화 후 Transformer에 입력, [CLS] 토큰으로 분류. 대표적 확장으로 DETR(객체 검출용 Transformer)이 있음.

---
*출처: 컴퓨터비전 개론 11, 16차시 / CV응용 심화 10, 13(의료영상), 14차시 재구성*
