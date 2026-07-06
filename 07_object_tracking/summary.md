# 추적 알고리즘 (Object Tracking)

## 1. 동영상 읽기 & 쓰기 (사전 준비)

| 항목 | 코드 |
|---|---|
| VideoCapture | `cap = cv2.VideoCapture(0)` (0=웹캠, 경로=파일) |
| 프레임 읽기 | `ret, frame = cap.read()` → ret: 성공 여부 |
| VideoWriter | `cv2.VideoWriter(path, fourcc, fps, (w,h))` |
| 릴리즈 | `cap.release()` / `out.release()` → 반드시 호출 |

## 2. 광학 흐름 (Optical Flow)

| 항목 | 내용 |
|---|---|
| 정의 | 연속 프레임 간 픽셀/객체의 겉보기 이동 벡터 추정 |
| Lucas-Kanade | 희소(Sparse) 광학 흐름: 특징점만 추적, `cv2.calcOpticalFlowPyrLK` |
| Farneback | 밀집(Dense) 광학 흐름: 전체 픽셀 추적, `cv2.calcOpticalFlowFarneback` |
| 가정 | 밝기 일정성(Brightness Constancy) + 공간적 일관성 |
| 피라미드 | 이미지 피라미드로 큰 움직임도 처리(Multi-scale) |

## 3. Meanshift & CamShift

| 알고리즘 | 원리 | 특징 | 한계 |
|---|---|---|---|
| Meanshift | 히스토그램 역투영 확률 맵에서 밀도 최대점으로 윈도우 이동 | 조명 변화에 강건 | 창 크기 고정, 스케일 변화 대응 불가 |
| CamShift | Meanshift 확장: 매 프레임 후 윈도우 크기·회전 자동 조정 | 스케일 변화 대응 가능 | 배경 색상과 유사할 때 실패 |

---
*출처: CV응용 심화 03, 04차시 재구성*
