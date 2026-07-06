# 데이터 전처리 (Data Preprocessing)

## 1. NumPy 기초 (배열/행렬 연산)

| 연산 | 코드 | 설명 |
|---|---|---|
| 배열 생성 | `np.zeros((3,4))`, `np.ones`, `np.arange` | 지정 형태의 초기 배열 |
| 형태 변환 | `a.reshape(2,-1)`, `a.flatten()` | -1은 자동 계산 |
| 행렬 곱 | `np.dot(A,B)` 또는 `A @ B` | 내적/행렬 곱 |
| 브로드캐스팅 | `A + b` (형태 자동 확장) | 스칼라·벡터·행렬 혼합 연산 |
| 축 연산 | `np.sum(axis=0)`, `np.mean(axis=1)` | 행/열 방향 집계 |

> 신경망 순전파는 본질적으로 행렬 곱의 연속: 입력 X(N×D) × 가중치 W(D×H) → 출력 Z(N×H). 배치 처리로 N개 샘플을 한 번에 병렬 계산.

## 2. OpenCV 기반 이미지 처리 기초

| 항목 | 함수 |
|---|---|
| 이미지 로딩 | `cv2.imread(path)` → **BGR** 순서로 로드 (RGB 아님 주의!) |
| 색공간 변환 | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB / BGR2GRAY / BGR2HSV)` |
| 이미지 저장 | `cv2.imwrite(path, img)` |
| 크기 조정 | `cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)` |

### 2.1 필터링 & 엣지 검출
| 기법 | 함수 | 원리/특징 |
|---|---|---|
| 가우시안 블러 | `cv2.GaussianBlur(img,(k,k),σ)` | 노이즈 제거, σ 클수록 많이 흐림 |
| 미디언 블러 | `cv2.medianBlur(img, k)` | 소금-후추 노이즈에 효과적 |
| Sobel | `cv2.Sobel(img, ddepth, dx, dy)` | 수평/수직 기울기 계산 |
| Canny | `cv2.Canny(img, low, high)` | 비최대 억제+이력 임계값, 정확한 엣지 |
| Laplacian | `cv2.Laplacian(img, ddepth)` | 2차 미분, 엣지 방향 무관 |
| 모폴로지 | `cv2.erode` / `cv2.dilate` | 침식(작은 객체 제거), 팽창(구멍 메움) |

### 2.2 색공간(Color Space)
| 색공간 | 채널 구성 | 주요 활용 |
|---|---|---|
| BGR/RGB | Blue-Green-Red | 표준 컬러 이미지 처리 |
| Grayscale | 단일 밝기 채널 | 이진화, 엣지 검출 전처리 |
| HSV | Hue-Saturation-Value | 색상 기반 객체 마스킹(피부색, 신호등) |

> ⚠️ OpenCV는 imread 시 BGR 순서로 저장. Matplotlib 시각화 시 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 변환 필수!

## 3. 기하학적 변환 (Geometric Transformation)

### 3.1 어파인(Affine) 변환
| 항목 | 내용 |
|---|---|
| 정의 | 평행성과 직선성 유지(6자유도: 이동, 회전, 스케일, 전단) |
| 행렬 형태 | 2×3 변환 행렬 M → `dst = M · [x,y,1]ᵀ` |
| 회전 | `cv2.getRotationMatrix2D(center, angle, scale)` |
| 함수 | `cv2.warpAffine(img, M, (W,H))` |

### 3.2 원근(Perspective) 변환
| 항목 | 내용 |
|---|---|
| 정의 | 평행성 비유지, 사영 기하학(8자유도) → 3×3 행렬 |
| 활용 | 문서 스캔 교정, 차선 Bird's Eye View 변환 |
| 계산 | `cv2.getPerspectiveTransform(src_pts, dst_pts)` (4점 대응) |
| 적용 | `cv2.warpPerspective(img, M, (W,H))` |

> ⚠️ 4쌍의 대응점 순서: 좌상-우상-좌하-우하 일관성 유지 필수.

### 3.3 이미지 보간법(Interpolation)
| 보간법 | 특징 | 권장 사용 |
|---|---|---|
| INTER_NEAREST | 최근접 픽셀, 속도 최고 | 이진 마스크 축소 |
| INTER_LINEAR | 쌍선형, 속도·품질 균형(기본값) | 일반 축소/확대 |
| INTER_CUBIC | 쌍삼차, 부드러운 결과 | 사진 확대 |
| INTER_LANCZOS4 | Lanczos 4×4 커널, 최고 품질 | 고품질 업샘플링 |
| INTER_AREA | 영역 평균, 다운샘플 특화 | 이미지 축소 |

## 4. 손실함수/정규화와 연결되는 전처리 원칙
- 데이터 정규화(Standardization): `(x−μ)/σ` — 특성 스케일이 다르면 학습 불안정
- 텐서 ↔ NumPy 배열 변환: `tensor.numpy()` / `torch.from_numpy()` (GPU 텐서는 `.cpu()` 먼저 호출)

---
*출처: 컴퓨터비전 개론 02차시 / CV응용 심화 01, 02차시 재구성*
