# SSD (Object Detection)

객체 검출 전반(Two-stage → Single-stage → SSD)의 흐름으로 정리.

## 1. Two-stage Detection: R-CNN 계열 발전 과정

| 모델 | 연도 | 핵심 혁신 | 속도 |
|---|---|---|---|
| R-CNN | 2014 | Selective Search 후보 → CNN → SVM 분류 | 느림(47초/장) |
| Fast R-CNN | 2015 | Feature Map에서 RoI Pooling → CNN 공유 | 빠름(2초/장) |
| Faster R-CNN | 2015 | RPN(Region Proposal Network)으로 완전 통합 | 실시간(0.2초) |
| Mask R-CNN | 2017 | Faster R-CNN + RoI Align + Mask Head | 실시간 + 분할 |

### Faster R-CNN 상세 구조
| 구성요소 | 설명 |
|---|---|
| Backbone | VGG16 / ResNet → 공유 특징맵 추출 |
| RPN | Anchor 기반 후보 영역(ROI) 생성: cls(전경/배경) + reg(좌표 보정) |
| Anchor | 면적 3종(128,256,512) × 비율 3종(1:1,1:2,2:1) = 9가지 |
| RoI Pooling | 임의 크기 RoI → 고정 크기 특징맵(예: 7×7) 변환 |
| NMS | 비최대 억제: IoU>임계값 중복 박스 제거 |
| Loss | 분류 CE Loss + 경계박스 회귀 Smooth-L1 Loss 합산 |

## 2. Single-stage Detection: YOLO 계열

| 버전 | 핵심 혁신 | 속도/정확도 |
|---|---|---|
| YOLOv1(2016) | 격자 기반 1회 순전파 검출, 7×7 그리드 | 빠름, 소형 객체 약점 |
| YOLOv3(2018) | FPN 다중 스케일(3개 출력), DarkNet53 Backbone | 균형 잡힌 성능 |
| YOLOv5(2020) | PyTorch 구현, AutoAnchor, CSP Backbone | 실용적, 커뮤니티 큼 |
| YOLOv8(2023) | Anchor-free, Decoupled Head, C2f 모듈 | 고성능, 세그멘테이션 지원 |
| YOLOv10(2024) | NMS-free 이중 할당, 실시간 엔드투엔드 | 최고 속도·정확도 균형 |

### Two-stage vs Single-stage 비교
| 항목 | Two-stage(Faster R-CNN) | Single-stage(YOLO/SSD) |
|---|---|---|
| 처리 방식 | 후보 생성(RPN) → 분류/회귀 | 단일 순전파 |
| 속도 | 상대적으로 느림 | 빠름(실시간 적합) |
| 정확도 | 높음(특히 소형 객체) | 보통(개선 중) |
| 구조 복잡도 | 높음 | 낮음 |
| 주요 활용 | 의료영상, 정밀 분석 | 자율주행, 실시간 CCTV |

## 3. SSD (Single Shot MultiBox Detector)

| 항목 | 내용 |
|---|---|
| 구조 | VGG16 Backbone + 여러 스케일의 Feature Map에서 각각 검출 |
| 다중 스케일 | 38×38(소), 19×19, 10×10, 5×5, 3×3, 1×1(대) 6개 출력 |
| Anchor | 각 그리드 셀마다 비율/크기가 다른 기본 박스(Default Box) 사용 |
| Loss | Multibox Loss = Localization(Smooth-L1) + Confidence(CE) |
| Hard Neg | Hard Negative Mining: 음성(배경):양성 = 3:1 비율 유지 |

### FPN (Feature Pyramid Network)
| 항목 | 내용 |
|---|---|
| 문제 | 단일 스케일 특징맵 → 소형 객체 검출 어려움 |
| 아이디어 | Bottom-up(Backbone) + Top-down(업샘플) + Lateral Connection |
| 장점 | 고수준 의미 + 고해상도 정보를 모든 스케일에 전달 |
| PANet | FPN + Bottom-up Path Augmentation → 더 강한 특징 흐름 |

## 4. 공통 평가 지표

| 지표 | 설명 |
|---|---|
| IoU | 예측 박스 ∩ 정답 박스 / 예측 박스 ∪ 정답 박스, IoU>0.5면 TP |
| AP | Precision-Recall 곡선 아래 면적(클래스별) |
| mAP | 전체 클래스 AP 평균: COCO는 IoU 0.5~0.95 평균 |

---
*출처: CV응용 심화 11, 12, 13차시 재구성*
