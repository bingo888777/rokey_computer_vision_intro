# LSTM (이미지 캡셔닝: CNN + LSTM)

## 1. 기본 구조 (Encoder-Decoder)

| 항목 | 내용 |
|---|---|
| Encoder | 사전학습 CNN(ResNet/VGG) → 이미지 특징 벡터 추출 |
| Context Vector | CNN 마지막 FC 또는 Global AvgPool 출력 → LSTM 초기 은닉상태 |
| Decoder | LSTM/GRU → 이전 단어 임베딩 + Context 벡터 → 다음 단어 예측 |
| 학습 | Teacher Forcing: 학습 시 실제 정답 단어를 다음 입력으로 사용 |
| 추론 | Greedy Search 또는 Beam Search로 단어 순차 생성 |

## 2. Attention 메커니즘 (Bahdanau, 2015)

| 항목 | 내용 |
|---|---|
| 문제 | 긴 시퀀스에서 단일 Context 벡터로는 정보 압축에 한계 |
| 아이디어 | 디코더 각 스텝에서 인코더 출력 전체를 동적으로 가중합 |
| 에너지 계산 | e_tj = f(s_{t-1}, h_j) → 유사도 스코어 |
| 어텐션 가중치 | α_tj = softmax(e_tj) → 합=1 |
| Context 벡터 | c_t = Σ α_tj · h_j → 매 스텝 다른 문맥 사용 |
| 효과 | 어떤 이미지 영역에 집중하는지 시각화 가능(해석성↑) |

## 3. 평가 지표

| 지표 | 원리 | 범위 |
|---|---|---|
| BLEU | n-gram 정밀도 기반(1~4-gram 겹침) | 0~1 (1에 가까울수록 좋음) |
| ROUGE-L | 최장 공통 부분 수열(LCS) 기반 | 0~1 |

---
*출처: CV응용 심화 08차시 재구성*
