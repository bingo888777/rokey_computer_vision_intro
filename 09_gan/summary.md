# GAN & VAE (이미지 생성형 AI)

## 1. GAN (Generative Adversarial Network)

| 항목 | 내용 |
|---|---|
| 구조 | Generator(G) vs Discriminator(D) 적대적 학습 |
| G 목표 | D를 속이는 가짜 이미지 생성: min log(1−D(G(z))) |
| D 목표 | 진짜/가짜 구분: max [log D(x) + log(1−D(G(z)))] |
| Nash 균형 | G(z) 분포 ≈ 실제 데이터 분포가 될 때 D = 0.5로 수렴 |
| Mode Collapse | G가 일부 패턴만 반복 생성 → 다양성 저하 |
| 학습 불안정 | D가 너무 강하면 G 학습 불가, 균형 유지 필요 |

## 2. 주요 GAN 변형

| 모델 | 핵심 혁신 | 특징 |
|---|---|---|
| DCGAN | 완전 연결층 제거, Conv/BatchNorm 사용 | 안정적 학습, 고해상도 |
| Pix2Pix | 이미지→이미지 변환, L1 Loss + GAN Loss | 페어 데이터 필요 |
| CycleGAN | 비페어 데이터 변환, 순환 일관성 손실(Cycle Loss) | 도메인 변환(말→얼룩말 등) |
| StyleGAN | 스타일 기반 Generator, AdaIN, 점진적 성장 | 고품질 얼굴 생성 |

## 3. VAE (Variational AutoEncoder)

| 항목 | 내용 |
|---|---|
| 구조 | Encoder → 잠재 공간(μ, σ) → 샘플링 → Decoder |
| 잠재 공간 | z ~ N(μ, σ²) → 가우시안 분포로 정규화 |
| 재파라미터화 | z = μ + σ·ε (ε~N(0,1)) → 역전파 가능하게 |
| ELBO Loss | L = 재구성 손실 + KL 발산(잠재 분포↔표준정규 차이) |

## 4. GAN vs VAE

| | GAN | VAE |
|---|---|---|
| 이미지 품질 | 선명함 | 흐릿한 경향 |
| 학습 안정성 | 불안정 | 안정적 |

---
*출처: CV응용 심화 07차시 재구성*
