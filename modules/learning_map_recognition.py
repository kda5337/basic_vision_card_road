
"""
CardRoad - 맵 인식 모듈
담당: 김단아

[역할]
웹캠 영상에서 맵을 인식하고 다음 정보를 반환합니다.

- 맵의 행과 열 개수
- 장애물 배치
- 시작점 및 도착점
- 격자 외곽의 네 꼭짓점 좌표

[인터페이스]
입력: OpenCV 이미지 (np.ndarray)
출력: DetectionResult[MapData]

구체적인 인식 알고리즘은 담당자가 자유롭게 구현합니다.
"""

"""CardRoad 맵 인식 모듈. numpy와 opencv-python이 필요합니다.

프로젝트 공통 모델:
    MapData(rows, cols, grid, start, goal, corners)
    DetectionResult(data=None, error_code=None): result.ok로 성공 여부 확인.
grid: 2차원 정수 리스트. 0은 이동 가능, 1은 장애물입니다.
start/goal: 0부터 시작하는 (행, 열). 미검출이면 None입니다.
MapData의 start와 goal 타입은 tuple[int, int] | None이어야 합니다.
corners: 원본 이미지 좌표. 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 순서입니다.
밝은 종이에 진한 선으로 그린 격자를 대상으로 합니다.
초록색은 시작점, 빨간색은 도착점, 검게 채운 칸은 장애물입니다.
손이나 물체에 가려진 격자는 정확한 인식을 보장하지 않습니다.
"""

import cv2
import numpy as np

from common.models import MapData, DetectionResult
from pathlib import Path


WARP_SIZE = 720
MAX_CELLS = 40
_last_recognition_method = None


class _RecognitionError(Exception):
    pass


def _failure(code):
    return DetectionResult(data=None, error_code=code)


def _success(rows, cols, obstacles, start, goal, corners):
    return DetectionResult(
        data=MapData(
            rows=rows, cols=cols, grid=obstacles.astype(np.uint8).tolist(),
            start=start, goal=goal, corners=np.asarray(corners, dtype=np.float32),
        ),
        error_code=None,
    )

'''
사각형의 네 꼭짓점을 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 순서로 정렬하는 함수
맵을 정면에서 본 모양으로 보정할 때, 사진의 각 모서리를 어느 모서리에 대응시킬지 정하기 위해 사용

1. reshape(4, 2)로 네 점을 각각 (x, y) 좌표로 정리.
2. mean(axis=0)으로 네 점의 중심을 구하고, arctan2()로 중심에서 각 점을 바라보는 각도를 계산.

dx = points[:, 0] - center[0]  # 중심에서 각 꼭짓점까지의 가로 이동량
dy = points[:, 1] - center[1]  # 중심에서 각 꼭짓점까지의 세로 이동량
angles = np.arctan2(dy, dx)    # 각 꼭짓점이 어느 방향에 있는지 각도로 계산

3. argsort(angles)로 각도순으로 정렬. 이미지에서는 아래로 갈수록 y가 커지므로 점들이 시계 방향으로 나열.
4. np.roll()로 목록을 돌려 x + y가 가장 작은 점을 맨 앞에 두기. 대체로 똑바로 놓인 맵에서는 이 점이 왼쪽 위.
즉, 각도는 네 점을 도는 방향을 정하고, x + y는 어디서 시작할지 정해. 여기서 4는 맵의 행 수와 관계없이 바깥 꼭짓점의 개수.
'''
def _order_quad(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0).copy()

'''
사진에서 맵의 바깥 테두리일 만한 사각형을 찾아 후보 목록으로 반환하는 함수
손으로 그린 격자의 바깥 테두리도, 노트 종이 자체의 테두리도 후보가 될 수 있음
어느 후보 안에 실제 격자가 있는지는 다음 단계인 _geometry()에서 확인
'''
def _quad_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) # 여기까지 사진 전처리

    binary = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9,) # 주변보다 어두운 펜 자국을 강조
    edges = cv2.Canny(gray, 40, 130) # 밝기가 급격히 바뀌는 경계를 찾기, Canny = 밝기가 급격히 변하는 경계 픽셀을 찾는 함수
                                    # 40과 130은 낮은 임계값과 높은 임계값. 
                                    # 강한 경계는 남기고, 약한 경계는 강한 경계와 연결된 경우에 남기기
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) # MORPH_CLOSE는 경계선의 작은 틈을 이어주기
                                    # cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ...)
                                    # 여기서는 닫힘 연산을 수행. 흰 경계선을 잠시 굵게 만들었다가 다시 줄여서, 
                                    # 가까운 경계선 사이의 작은 틈을 메우기. 이 코드에서는 3×3 커널을 사용.

    '''
    findContours(), contourArea()	
    두 영상에서 각각 윤곽선을 찾고, 윤곽이 둘러싼 면적이 전체 사진의 2.5% 미만이면 건너뛰기.

    cv2.findContours(mask, ...) = 흑백 영상에서 흰 영역의 테두리를 따라가며 윤곽선별 좌표 목록을 만들기. 
                                위의 Canny의 결과는 경계가 표시된 이미지이고, findContours의 결과는 각 테두리를 이루는 점들의 목록.
    cv2.contourArea(contour) = 	윤곽선이 둘러싼 면적을 계산. 펜으로 칠한 픽셀 수가 아니라, 테두리 안쪽이 차지하는 넓이. 단위는 픽셀².

    왜 전체 사진 면적의 2.5%보다 작은 윤곽을 건너뛸까?
    글씨, 펜 자국, 격자 한 칸의 테두리처럼 작은 윤곽도 찾음. 
    이들을 전부 사각형으로 근사하면 검사량이 늘고, 격자 한 칸이 맵 전체의 후보로 들어올 수 있음. => 때문에 너무 작으면 무시
    멀리서 찍어서 실제 맵도 사진의 2%밖에 차지하지 않으면, 이 단계에서는 실제 맵까지 제외될 수 있음.
    '''
    min_area = frame.shape[0] * frame.shape[1] * 0.025
    candidates = []
    for mask in (binary, edges):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours:
            # contour 한 개는 점 한 개가 아니라, 테두리 하나를 따라가며 모은 여러 점의 좌표 배열
            if abs(cv2.contourArea(contour)) < min_area:
                continue

            '''
            approxPolyDP(), isContourConvex()	
            울퉁불퉁한 윤곽을 단순한 다각형으로 근사. 
            꼭짓점이 정확히 4개이고 안쪽으로 움푹 들어간 부분이 없는 볼록 사각형만 남기기.

            cv2.arcLength(contour, True) = 윤곽선의 둘레 길이를 픽셀 단위로 구하기. 
                                            True는 마지막 점에서 첫 점까지 이어진 닫힌 윤곽으로 계산한다는 뜻.
            cv2.approxPolyDP(contour, 오차, True) = 울퉁불퉁한 윤곽을 더 적은 꼭짓점으로 이루어진 다각형으로 단순화. 
                                                허용 오차가 크면 작은 굴곡을 더 많이 생략할 수 있음.
            
            왜 0.015, 0.025, 0.04를 차례로 쓸까?
            세 숫자는 윤곽의 둘레에 곱하는 허용 오차 비율. 
            둘레가 1,000픽셀이라면 approxPolyDP()에 전달되는 실제 허용 오차는 각각 15픽셀, 25픽셀, 40픽셀.
                                    B                 ← 손으로 그린 윗변의 튀어나온 점
                          /│\
                        /  │  \
                      /    │    \
                    /      │      \
                  /        │ 20px   \
                A··········┴··········C      ← A와 C를 바로 이은 가상의 직선
                │                     │
                │                     │
                E─────────────────────D

                                | 허용 오차 15픽셀 | 허용 오차 25픽셀 |
            | B를 생략하면?     | 원래 선과 20픽셀차이 | 원래 선과 20픽셀차이 |
            | 허용 범위 안일까? | `20 > 15` → 너무 많이 달라져 | `20 ≤ 25` → 이 정도 차이는 허용해 |
            | 단순화한 윤곽     | `A → B → C → D → E` | `A → C → D → E` |
            | 결과              | B를 남겨서 **꼭짓점 5개** | B를 생략해서 **꼭짓점 4개** |
            '''
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.015, 0.025, 0.04):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    #  convex면 오목하니까 skip해야 되고, 면이 4개가 아니면 사각형이 아니니까 skip
                    continue

                '''
                _order_quad(), np.linalg.norm()	
                꼭짓점 순서를 맞춘 뒤, 가장 짧은 변이 35픽셀 미만이면 제외. 
                기존 후보와 대응하는 네 꼭짓점의 평균 거리가 8픽셀 미만이면 같은 후보로 보기.
                np.linalg.norm() = 벡터의 길이를 구하는 함수, 두 점의 좌표 차이를 넣어서 두 점 사이의 직선거리를 계산

                이전_꼭짓점 = np.roll(quad, 1, axis=0)                 # [A, B, C, D] → [D, A, B, C]
                네_변의_길이 = np.linalg.norm(quad - 이전_꼭짓점, axis=1)

                if np.min(네_변의_길이) < 35:
                    continue

                quad는 방금 찾은 사각형이고, old는 candidates에 이미 저장된 사각형을 하나씩 꺼내 부르는 이름. 
                for old in candidates에서 정해지는 변수라서 앞에서 따로 선언할 필요가 없음.
                기존 사각형 중 어느 것과 비교해도 평균 거리가 8픽셀 이상이면 if not any(...)가 참이 되고, 
                candidates.append(quad)가 실행    
                '''
                quad = _order_quad(polygon)
                if np.min(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) < 35:
                    continue
                if not any(np.mean(np.linalg.norm(quad - old, axis=1)) < 8
                           for old in candidates):
                    candidates.append(quad)
                break
    '''
    orted(..., reverse=True)[:30]	
    면적이 큰 순서로 최대 30개를 반환. 후보 하나에는 네 꼭짓점의 (x, y) 좌표가 들어 있음.
    '''
    return sorted(candidates, key=cv2.contourArea, reverse=True)[:30]

'''
격자선이 어느 좌표에 있는지 찾아서, 선 하나당 좌표 하나를 반환하는 함수
_geometry()에서 이 함수를 두 번 호출. 한 번은 세로선의 x 좌표를 찾고, 한 번은 가로선의 y 좌표를 찾기
왜 필요할까?
눈으로 보면 두께가 3픽셀인 검은 선도 격자선 하나임. 
그런데 이미지에서는 그 선이 x = 99, 100, 101 세 열에 걸쳐 있음(3픽셀크기의 선이니까). 
세 열을 각각 선으로 세면 칸 개수가 틀려짐. 
이 함수는 이웃한 픽셀들을 한 줄로 묶어서 x ≈ 100이라는 대표 좌표 하나를 구한다. 
그래야 다음 단계에서 “첫 번째 칸은 x = 100과 x = 200 사이”라고 판단할 수 있음.

mask = 원근 보정 후 이진화한 2차원 이미지. 이 코드에서는 원본 사진의 어두운 격자선이 마스크에서 흰색(255), 배경이 검은색(0).
vertical=True, 세로선을 찾기. 세로선이 왼쪽에서 얼마나 떨어졌는지를 나타내는 x 좌표 배열을 반환.
vertical=False, 가로선을 찾기. 가로선이 위에서 얼마나 떨어졌는지를 나타내는 y 좌표 배열을 반환.
positions = 검출한 선들의 중심 위치를 작은 값부터 담은 1차원 np.float32 배열. 
            좌표는 원근 보정된 이미지 기준. 검증에 실패하면 배열 대신 _RecognitionError 예외가 발생.
'''
def _line_positions(mask, vertical):
    length = max(15, mask.shape[0 if vertical else 1] // 6) # length는 이미지 높이의 1/6로 정하되 최소 15픽셀
    kernel = cv2.getStructuringElement( # cv2.getStructuringElement() = 영상에서 특정 모양을 검사할 작은 배열인 커널을 만드는 함수
                        #(1, length)를 주면 세로로 긴 커널, (length, 1)을 주면 가로로 긴 커널을 만듬. 크기 순서는 (너비, 높이).

        cv2.MORPH_RECT, (1, length) if vertical else (length, 1)) # cv2.MORPH_RECT = 커널의 모양을 직사각형으로 지정하는 상수	
                    # getStructuringElement()의 첫 번째 인자로 넣음. 함수처럼 실행되는 게 아니라 “이 모양으로 만들어 줘”라는 선택값.

    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # cv2.morphologyEx() = 커널을 이미지에 적용해 형태 연산을 수행하는 함수
                        # 흰 영역을 먼저 침식한 다음 다시 팽창시켜서, 커널 방향으로 충분히 긴 획은 남기고 짧은 획이나 작은 점은 제거.
    '''
    vertical=True로 세로선을 찾는 경우를 그림으로 보면 이래. 
    █는 흰 픽셀(255), ·는 검은 픽셀(0).
    입력 mask                  결과 lines
    ··█·····█··                ··█·····█··
    ··█·····█··                ··█·····█··
    ███████████                ··█·····█··
    ··█·····█··                ··█·····█··
    ··█··█··█··      ──▶      ··█·····█··
    ··█·····█··                ··█·····█··
    ███████████                ··█·····█··
    ··█·····█··                ··█·····█··
    ··█·····█··                ··█·····█··

    원본 mask       침식 후        다시 팽창한 후
    ·  ·            ·  ·          ·  ·
    █  ·            ·  ·          █  ·
    █  ·            █  ·          █  ·
    █  █            █  ·          █  ·
    █  ·            █  ·          █  ·
    █  ·            ·  ·          █  ·
    ·  ·            ·  ·          ·  ·
    흰 영역을 침식한다는 건, 흰 픽셀마다 “이 픽셀을 중심으로 커널을 놓았을 때, 커널 전체가 흰색 안에 들어가는가?”를 검사하는 것
    전부 흰색이면 그 자리를 흰색으로 남기고, 한 칸이라도 검은색이면 그 자리를 검은색으로 바꾼다. 
    팽창은 침식 후 남은 흰 픽셀을 기준으로 위아래 한 칸씩 다시 넓힌다
    '''
    
    profile = (lines > 0).mean(axis=0 if vertical else 1)
    indices = np.flatnonzero(profile >= 0.40) # np.flatnonzero() = 배열에서 0이 아닌 값의 인덱스를 반환하는 함수
    '''
    흰 픽셀의 비율을 확인하는 이유는 그 열에 흰 점이 조금 찍힌 것인지, 세로로 길게 이어진 격자선이 있는지 구분하기 위해서야.

    예를 들어 높이가 100픽셀인 이미지에서 세로선을 찾는다고 해보자.
    검사한 열	흰 픽셀 수	흰 픽셀 비율	코드의 판단
    x = 20	    5개	            5%	        짧은 자국일 수 있으므로 제외
    x = 50	    80개	        80%	        세로 격자선 후보로 선택
    두 열 모두 “흰 픽셀이 있는가?”라고 물으면 답은 같음. 
    하지만 “이 열의 얼마만큼이 흰색인가?”라고 물으면 x = 50이 세로선의 위치일 가능성이 더 크다고 판단할 수 있음.
    '''

    if len(indices) == 0:
        raise _RecognitionError('GRID_NOT_FOUND')
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) 
    # np.diff() = 배열에서 바로 다음 값에서 현재 값을 뺀 차이를 순서대로 구하는 함수
    # np.diff(indices) > 1은 연속된 픽셀 좌표가 어디서 끊기는지 찾는 조건. 
    # 10 → 11 → 12는 차이가 1이라 같은 묶음이고, 12 → 30은 차이가 18이라 30부터 새 묶음이 시작돼.
    # 차이가 1이라는 건 픽셀 좌표가 바로 옆에 붙어 있다는 뜻

    # 넓게 채워진 영역은 격자선 후보에서 제외합니다.
    positions = np.array([
        np.average(group, weights=profile[group])
        for group in groups if len(group) <= WARP_SIZE * 0.035 # 세로 기준 = 세로선들이 가로로 너무 두꺼운지 확인
    ], dtype=np.float32)
    if not 3 <= len(positions) <= MAX_CELLS + 1: # MAX_CELLS + 1, 현재는 최대 40개의 칸 최대칸 개수 조절, 이 부분이 격자인지 결정
                                                 # 격자이러면, 가로든 세로든 선이 무조건 3개 이상이여아 하니까
        raise _RecognitionError('GRID_NOT_FOUND')
    gaps = np.diff(positions)
    spacing = np.median(gaps) # spacing은 각 칸의 폭이 지나치게 좁거나 넓지 않은지 검사

    # 선이 누락되거나 추가되어 간격이 불규칙하면 인식 실패로 처리합니다.
    if spacing < 12 or np.max(np.abs(gaps / spacing - 1)) > 0.22:
        raise _RecognitionError('GRID_IRREGULAR')
    if positions[-1] - positions[0] < WARP_SIZE * 0.5: # positions[-1], 마지막 격자선의 좌표
        raise _RecognitionError('GRID_NOT_FOUND')
    return positions

'''
사진 속 사각형 후보를 정면에서 본 모습으로 펴고, 그 안에서 각 칸의 경계를 찾는 함수
전체 코드에서 _quad_candidates()가 “여기에 격자가 있을 것 같다”는 후보를 찾으면, 
_geometry()가 “어디부터 어디까지가 한 칸인가”를 구하기. 
그다음 _read_cells()가 각 칸을 잘라 _classify()에 넘기면 출발점·도착점·장애물·빈칸이 판별되는 식.

함수는 어떤 순서로 동작?
1. 사각형 후보를 720×720 이미지에 정면으로 폅니다.
   quad는 입력 사진에서 찾은 후보의 네 꼭짓점. cv2.getPerspectiveTransform(quad, destination)은 
   이 점들을 (12, 12), (707, 12), (707, 707), (12, 707)로 옮기는 변환 행렬 matrix를 구한다. 
   cv2.warpPerspective()가 그 변환을 사진에 적용한 결과가 warped.

2. 격자선을 찾기 위한 흑백 이미지 mask를 만듭니다.
   warped를 회색조로 바꾼 다음, adaptiveThreshold()로 각 픽셀의 밝기를 주변과 비교. 
   THRESH_BINARY_INV를 쓰므로 밝은 종이 위의 어두운 선은 마스크에서 흰색으로 나타내기. 
   이때 선을 찾을 때는 mask를 쓰고, 나중에 칸의 색을 읽을 때는 컬러 이미지 warped를 사용.

3. 세로선의 x좌표 xs와 가로선의 y좌표 ys를 구합니다.
   _line_positions(mask, True)는 세로선을, _line_positions(mask, False)는 가로선을 찾는다. 
   예를 들어 세로선이 5개이고 가로선이 4개라면, 칸은 선과 선 사이이므로 3행 4열. 
   이 단계에서 선이 부족하거나 간격이 너무 불규칙하면 오류를 발생시키고, detect_map()은 다음 사각형 후보를 검사.

4. 검출한 격자의 바깥 꼭짓점을 사진 좌표로 되돌립니다.
   xs와 ys의 첫 값과 마지막 값을 조합하면 보정 이미지에서의 격자 네 모서리 outer가 나온다. 
   여기에 역변환 행렬 np.linalg.inv(matrix)를 적용한 결과가 corners. 
   처음 받은 quad는 검사를 시작한 후보의 모서리이고, corners는 실제로 찾은 가장 바깥 격자선을 기준으로 계산한 모서리이므로 서로 다를 수 있음.
마지막 return warped, xs, ys, corners에서 warped, xs, ys는 칸을 잘라 읽는 데 쓰이고, 
corners는 인식 결과를 원본 사진 위에 다시 그리는 데 쓰암. 
정확히는 _geometry()가 작업용 이미지 working의 좌표로 corners를 반환하고, 
사진을 축소해서 처리한 경우 detect_map()이 coordinate_scale을 곱해 원본 좌표로 맞춘다..
'''
def _geometry(frame, quad):
    n = WARP_SIZE
    # 바깥 격자선이 영상 경계에 닿지 않도록 여백을 둡니다.
    destination = np.array([[12, 12], [n-13, 12], [n-13, n-13], [12, n-13]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(frame, matrix, (n, n), borderValue=(255, 255, 255))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9,)
    
    xs = _line_positions(mask, True)
    ys = _line_positions(mask, False)
    outer = np.array([
        [xs[0], ys[0]], [xs[-1], ys[0]],
        [xs[-1], ys[-1]], [xs[0], ys[-1]],
    ], np.float32) # [0]은 배열의 첫 값이고 [-1]은 마지막 값. 
                # 예를 들어 xs[-1]은 맨 오른쪽 세로선, ys[0]은 맨 위 가로선이므로, 둘을 짝지은 [xs[-1], ys[0]]은 오른쪽 위 꼭짓점.
    corners = cv2.perspectiveTransform(outer[None], np.linalg.inv(matrix))[0] # corners에는 격자판의 바깥 꼭짓점 4개를 
                                                        # 보정 전 사진에서 찾았을 때의 (x, y) 픽셀 좌표가 들어간다. 
                                                        # outer와 corners는 같은 네 점을 가리키고, 좌표를 읽는 이미지가 다름.
    return warped, xs, ys, corners

'''
_classify(cell)은 격자 한 칸의 안쪽을 잘라낸 컬러 이미지를 받아, 
그 칸이 'start'(출발점), 'goal'(도착점), 'obstacle'(장애물), 'empty'(빈칸) 중 무엇인지 판별하는 함수. 
판정하기 애매하면 CELL_AMBIGUOUS 오류를 발생.
'''
def _classify(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # 어두운 초록 펜도 포함하되 회색 펜의 작은 채널 차이는 제외합니다.
    blue_ch, green_ch, red_ch = cv2.split(cell.astype(np.int16))
    green = ((hue >= 35) & (hue <= 85) & (saturation >= 45) & (value >= 35)
             & (green_ch - np.maximum(blue_ch, red_ch) >= 10))
    red = ((hue <= 12) | (hue >= 168)) & (saturation >= 85) & (value >= 65)
    green_ratio, red_ratio = green.mean(), red.mean()
    dark_ratio = ((value < 85) & ~green & ~red).mean()
    # 가는 초록 펜 획을 위해 셀 안쪽 면적의 3%를 기준으로 합니다.
    has_green = green_ratio >= 0.03
    if has_green and red_ratio >= 0.045: # 초록색이 3% 이상이고 빨간색도 4.5% 이상 = CELL_AMBIGUOUS 오류
        raise _RecognitionError('CELL_AMBIGUOUS')
    if has_green:
        return 'start'
    if red_ratio >= 0.045:
        return 'goal'
    # 두 색 모두 기준 미만 = 장애물 검사로 진행
    # 셀 크기를 통일한 뒤 주변 종이보다 어두운 저채도 획의 면적을 잽니다.
    # 닫힘 연산은 가는 펜 자국을 메우므로 넓은 그림자와 빗금을 구분하는 데 쓰입니다.
    small_v = cv2.resize(value, (64, 64), interpolation=cv2.INTER_AREA)
    small_s = cv2.resize(saturation, (64, 64), interpolation=cv2.INTER_AREA)
    paper = cv2.morphologyEx(small_v, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    # 표식 색이 있는 픽셀은 검은 빗금의 면적에서 제외합니다.
    marker_fraction = cv2.resize((green | red).astype(np.float32), (64, 64),
                                 interpolation=cv2.INTER_AREA)
    stroke_ratio = ((cv2.subtract(paper, small_v) > 32) # 펜으로 빗금을 친 칸은 stroke_ratio로 찾는다. 
                    # 밝기 영상을 64×64로 맞춘 뒤, MORPH_CLOSE로 가느다란 검은 획을 메워 주변 종이의 밝기(paper)를 추정
                    & (small_s < 115) & (marker_fraction < 0.5)).mean()
    if dark_ratio >= 0.55 or stroke_ratio >= 0.22:
        return 'obstacle'
    # 색상 기반 추정이므로 피부색 물체에 가려진 칸을 빈칸으로 오인할 수 있습니다.
    unexpected_color = ((saturation > 100) & (value > 85)).mean()
    if dark_ratio > 0.15 or unexpected_color > 0.25:
        raise _RecognitionError('CELL_AMBIGUOUS')
    return 'empty'

'''
_read_cells(warped, xs, ys) = 보정된 격자판을 한 칸씩 읽어, 장애물 배치와 출발점·도착점의 (행, 열) 좌표를 만드는 함수
일반 맵 인식에서 _geometry()와 _classify()를 연결
_geometry(): 보정된 컬러 이미지 warped와 격자선 위치 xs, ys를 구한다.
_classify(cell): 잘라낸 한 칸을 보고 'start', 'goal', 'obstacle', 'empty' 중 하나를 반환.
_read_cells(): 어떤 칸을 잘랐는지 기억하고, 분류 결과를 지도의 해당 위치에 기록. 
예를 들어 _classify()가 'obstacle'을 반환했을 때, 그 장애물이 (1, 3)에 있다는 사실을 기록하는 곳이 _read_cells().
'''
def _read_cells(warped, xs, ys):
    # ① 격자선 개수로 행과 열의 수를 계산
    rows, cols = len(ys) - 1, len(xs) - 1
    obstacles = np.zeros((rows, cols), dtype=bool)
    starts, goals = [], []

    # ② 왼쪽 위 칸부터 한 행씩 돌면서 현재 칸의 경계를 찾는다.
    for row in range(rows):
        for col in range(cols):
            x0, x1 = xs[col:col+2] # x0, x1 = xs[2:4]이면 x0 = xs[2] ≈ 243.5, x1 = xs[3] ≈ 359.5
            y0, y1 = ys[row:row+2]
            mx, my = (x1-x0)*0.17, (y1-y0)*0.17 # 왼쪽과 오른쪽에서 폭의 17%씩, 위와 아래에서 높이의 17%씩 제외
                                                # 검은 격자선이 장애물 판정에 섞이는 것을 줄일 수 있음
            cell = warped[ 
                int(np.ceil(y0+my)):int(np.floor(y1-my)),
                int(np.ceil(x0+mx)):int(np.floor(x1-mx)),]
            if cell.size == 0:
                raise _RecognitionError('GRID_TOO_SMALL')
            label = _classify(cell)
            if label == 'obstacle':
                obstacles[row, col] = True
            elif label == 'start':
                starts.append((row, col))
            elif label == 'goal':
                goals.append((row, col))
    if len(starts) > 1:
        raise _RecognitionError('START_COUNT_INVALID')
    if len(goals) > 1:
        raise _RecognitionError('GOAL_COUNT_INVALID')
    return rows, cols, obstacles, starts[0] if starts else None, goals[0] if goals else None

'''
손그림 맵에서 검출된 선분들을 격자선별로 묶고, 각 격자선을 대표하는 직선 하나를 구하는 함수. 
_handdrawn_maps()가 이 함수를 두 번 호출. vertical=True일 때는 세로선 목록을, False일 때는 가로선 목록을 얻는다.

전체 코드에서 왜 필요한가요?
앞에서 본 _line_positions()는 정면으로 보정한 영상에서 세로선의 위치를 x = 120처럼 좌표 하나로 나타냈음. 
그런데 손으로 그린 세로선은 위쪽에서는 x = 120이고 아래쪽에서는 x = 130일 수 있음. 
또 펜 자국이 중간에 끊기면 하나의 격자선에서 여러 선분이 검출될 수 있음.

그래서 _stroke_lines()는 세로선을 x = a*y + b라는 직선식으로 표현. 
이렇게 하면 선이 기울어져 있어도 어느 높이에서든 x좌표를 계산할 수 있음. 
이후 _handdrawn_maps()가 이 세로선과 가로선의 교점을 구하고, 교점 네 개를 모아 한 셀의 위치를 정한다.
'''
def _stroke_lines(mask, vertical):
    """짧고 기울어진 선분들을 묶어 손그림 격자선을 직선으로 근사합니다."""
    h, w = mask.shape
    span = h if vertical else w
    extent = w if vertical else h
    # ① 마스크에서 선분을 찾는다. 
    # cv2.HoughLinesP()가 펜 자국에서 직선처럼 이어진 부분을 찾아 양 끝점 (x1, y1, x2, y2)으로 반환. 
    # 이때 찾은 것은 아직 개별 선분. 같은 격자선의 위쪽과 아래쪽이 서로 다른 선분으로 나올 수 있음. 
    # 선분이 하나도 없으면 GRID_NOT_FOUND 오류.
    segments = cv2.HoughLinesP( mask, 1, np.pi / 360, threshold=max(20, int(span * .10)),
        minLineLength=max(25, int(span * .22)), maxLineGap=max(8, int(span * .08)),)
    if segments is None or not segments.size:
        raise _RecognitionError('GRID_NOT_FOUND')
    segments = np.asarray(segments).reshape(-1, 4)

    records = []
    for x1, y1, x2, y2 in segments:
        u1, v1, u2, v2 = (y1, x1, y2, x2) if vertical else (x1, y1, x2, y2) # 세로선을 찾을 때는 u = y, v = x
        if abs(u2-u1) < 1 or abs(v2-v1) > abs(u2-u1) * .35:
            continue # 무시하는 경우 ① abs(u2-u1) < 1 — 찾으려는 방향으로 전혀 뻗지 않은 경우
                     # ② abs(v2-v1) > abs(u2-u1) * .35 — 원하는 방향에서 너무 많이 기울어진 경우
        slope = (v2-v1) / (u2-u1)
        offset = v1 - slope*u1
        records.append((slope*span/2 + offset, u1, v1, u2, v2))
    records.sort()

    groups = []
    for record in records:
        if not groups or record[0] - np.median([r[0] for r in groups[-1]]) > max(5, extent*.035):
            groups.append([]) # 조건이 참일 때만 새 그룹 생성
        groups[-1].append(record)

    fitted = []
    for group in groups:
        u = np.array([v for r in group for v in (r[1], r[3])], float)
        v = np.array([v for r in group for v in (r[2], r[4])], float)
        if np.ptp(u) < span*.55: # np.ptp() = 배열에서 최댓값과 최솟값의 차이를 구하는 함수
            # span * .55 = 한 그룹에 모인 선분들이 격자선 후보로 인정받으려면 어느 정도 길게 뻗어 있어야 하는지 정하는 기준
            continue
        slope, offset = np.polyfit(u, v, 1) # np.polyfit(u, v, 1) = 같은 그룹에 모인 선분들의 끝점을 
                                            # 가장 잘 나타내는 직선 하나를 구하는 함수
        fitted.append((slope, offset))
    if not 3 <= len(fitted) <= MAX_CELLS+1:
        raise _RecognitionError('GRID_NOT_FOUND')
    positions = np.array([a*span/2+b for a, b in fitted])

    gaps = np.diff(positions)
    if np.min(gaps) < 12 or np.max(np.abs(gaps/np.median(gaps)-1)) > .45:
        raise _RecognitionError('GRID_IRREGULAR')
    if positions[-1]-positions[0] < extent*.65:
        raise _RecognitionError('GRID_NOT_FOUND')
    return fitted

'''
손그림에서 세로 방향으로 이어진 획과 가로 방향으로 이어진 획을 따로 추출하는 함수. 
입력 ink는 어두운 펜 자국을 흰색(255), 배경을 검은색(0)으로 나타낸 이진 이미지. 
출력은 같은 크기의 이미지 두 장, vertical_mask와 horizontal_mask

전체 코드에서 왜 필요한가?
손으로 그린 맵에서는 격자 테두리와 장애물 빗금이 모두 검은 펜 자국일 수 있음. 
예를 들어 한 칸 안에 /// 모양의 빗금이 있으면, 원래 ink에는 세로 테두리 |, 가로 테두리 ─, 빗금 /이 모두 흰색으로 나타남. 
이때 획의 방향을 이용하면 세로선 검출에는 세로에 가까운 획만, 가로선 검출에는 가로에 가까운 획만 넘길 수 있음.

이 함수는 일반 맵 인식과 첫 번째 손그림 인식 시도가 모두 실패했을 때, 
_handdrawn_maps(frame, directional=True)에서 호출. 
앞서 본 _stroke_lines()와 연결하면, _directional_strokes()는 방향별로 픽셀을 골라내고, 
_stroke_lines()는 그 픽셀에서 선분을 찾아 대표 직선식을 구한다..
'''
def _directional_strokes(ink):
    """가로·세로에 가까운 긴 획만 남겨 대각선 빗금과 격자를 분리합니다."""
    masks = []
    for vertical in (True, False):
        strokes = np.zeros_like(ink) # strokes: 매번 검은색으로 비워 둔 결과 이미지

        for tilt in range(-5, 6):
            kernel = np.zeros((33, 33), np.uint8)
            a, b = (16-tilt, 0), (16+tilt, 32)
            if not vertical:
                a, b = a[::-1], b[::-1]
            cv2.line(kernel, a, b, 1, 1) # cv2.line(): 커널 안에 두 점을 잇는 두께 1픽셀의 선을 그린다.
            strokes |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel) # MORPH_OPEN: 침식 후 팽창
            # 직관적으로는 흰 펜 자국 위에 선 모양 틀을 올려 보았을 때, 틀 전체가 들어맞는 곳의 획을 남기는 연산. 
            # 긴 세로획에는 세로 커널이 들어맞지만, 가느다란 45° 빗금에는 들어맞기 어렵다.

        masks.append(strokes)
    return masks

'''
사진 속 펜 자국에서 가로선과 세로선을 찾아 격자를 구성하고, 각 칸을 읽어 맵 후보를 만들어 내는 함수. 
앞에서 본 _directional_strokes(), _stroke_lines(), _classify()가 이 함수 안에서 연결.

전체 코드에서 왜 필요한가요?
일반 인식 경로는 _quad_candidates()로 격자 바깥의 사각형을 먼저 찾고, 그 사각형 전체를 정면으로 편 다음 칸을 읽음. 
그런데 손으로 그린 격자는 바깥 테두리가 중간에 끊겨 있거나, 선마다 조금씩 기울기가 달라서 이 방식으로 읽기 어려울 수 있음.

_handdrawn_maps()는 내부의 가로선과 세로선부터 찾는다. 
가로선과 세로선이 어디서 만나는지 계산하면, 바깥 사각형을 먼저 확정하지 않아도 각 칸의 네 모서리를 구할 수 있음. 
detect_map()은 일반 인식에서 성공한 후보가 없을 때 이 경로를 시도.
'''
def _handdrawn_maps(frame, directional=False):
    """일반 격자 검출 실패 시 끊기거나 조금 휘어진 진한 펜 선을 검출합니다.

    노트 줄과 펜 선을 분리하고, 연결 영역의 형태와 선 간격을 검사합니다.
    선을 직선으로 근사하므로 종이의 곡면 자체를 복원하지는 않습니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 9)

    # ① 사진에서 펜 자국을 추출
    # 사진을 회색조 gray로 바꾸고, 흐리게 만든 background와 비교. 
    # 주변 종이보다 충분히 어두운 픽셀을 골라 펜 자국이 흰색인 이진 이미지 ink를 만든다. 호출 방식에 따라 처리가 달라짐.
    # 첫 번째 시도에서 맵을 못 찾으면 directional=True, 밝기와 채도를 기준으로 잉크를 추출하고, 
    # _directional_strokes()로 세로획과 가로획을 분리.
    if directional:
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        ink = ((background.astype(np.float32)-gray > 25) & (gray < 166)
               & (saturation < 125)).astype(np.uint8)*255
        vertical_mask, horizontal_mask = _directional_strokes(ink)
        ink = vertical_mask | horizontal_mask
        gap = 15
    
    # 첫 번째 directional=False, 작은 잡음을 제거한 뒤, 같은 ink를 세로선 검출과 가로선 검출에 모두 사용.
    else:
        ink = ((background.astype(np.float32)-gray > 40)
               & (gray < 135) & (background > 140)).astype(np.uint8)*255
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        vertical_mask = horizontal_mask = ink
        gap = 13
    # ② 펜 자국이 모인 곳에서 격자 후보 영역을 찾는다.
    # MORPH_CLOSE로 가까이 끊어진 획을 이어 준 뒤, connectedComponentsWithStats()로 서로 연결된 픽셀 덩어리를 찾는다. 
    # 큰 덩어리부터 최대 20개를 살펴보고, 너무 작거나 지나치게 길쭉한 영역은 건너뛰기. 
    # 여기서 고른 영역은 아직 격자가 있을 만한 곳.
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected)

    # ③ 후보 영역 안에서 대표 가로선과 세로선을 구합니다.
    # _stroke_lines(..., True)는 세로선 목록 vertical을, _stroke_lines(..., False)는 가로선 목록 horizontal을 만든다. 
    # 각 선은 기울기와 절편으로 표현. 세로선 c의 식은 x = ax*y + bx, 가로선 r의 식은 y = ay*x + by.
    for component in sorted(range(1, count), key=lambda i: stats[i, 4], reverse=True)[:20]:
        x, y, w, h, area = stats[component]
        if min(w, h) < 70 or w*h < gray.size*.015 or area < 200:
            continue
        if not .2 < w/h < 5:
            continue
        region = labels[y:y+h, x:x+w] == component
        try:
            vertical = _stroke_lines(vertical_mask[y:y+h, x:x+w] * region, True)
            horizontal = _stroke_lines(horizontal_mask[y:y+h, x:x+w] * region, False)
            points = np.empty((len(horizontal), len(vertical), 2), np.float32)
            for r, (ay, by) in enumerate(horizontal):
                for c, (ax, bx) in enumerate(vertical):
                    # ④ 선들의 교점을 계산해 points에 저장합니다.
                    # 코드 속 px, py는 위 두 직선식을 동시에 만족하는 좌표. 즉, r번째 가로선과 c번째 세로선이 만나는 점
                    px = (ax*by+bx)/(1-ax*ay)
                    py = ay*px+by
                    if not (-w*.1 <= px <= w*1.1 and -h*.1 <= py <= h*1.1):
                        raise _RecognitionError('GRID_IRREGULAR')
                    # px, py는 잘라낸 후보 영역 안의 좌표이므로, 그 영역의 왼쪽 위 위치 (x, y)를 더해 입력 영상 기준 좌표로 바꾸기. 
                    # points[r, c]가 저장하는 것은 셀의 중심이 아니라 격자선의 교점
                    points[r, c] = (px+x, py+y)
            rows, cols = len(horizontal)-1, len(vertical)-1
            obstacles = np.zeros((rows, cols), bool)
            starts, goals = [], []
            destination = np.float32([[0, 0], [63, 0], [63, 63], [0, 63]])
            for r in range(rows):
                for c in range(cols):
                    cell_quad = np.array([points[r,c], points[r,c+1],
                                          points[r+1,c+1], points[r+1,c]], np.float32)
                    # 휘어진 경계선이 포함되지 않도록 셀 안쪽만 검사합니다.
                    center = cell_quad.mean(axis=0)
                    inner = center+(cell_quad-center)*.60
                    matrix = cv2.getPerspectiveTransform(inner, destination)
                    cell = cv2.warpPerspective(frame, matrix, (64, 64))
                    label = _classify(cell)
                    if label == 'obstacle':
                        obstacles[r,c] = True
                    elif label == 'start':
                        starts.append((r,c))
                    elif label == 'goal':
                        goals.append((r,c))
            if len(starts) > 1 or len(goals) > 1:
                raise _RecognitionError('MARKER_COUNT_INVALID')
            corners = np.array([points[0,0], points[0,-1], points[-1,-1], points[-1,0]], np.float32)
            yield rows, cols, obstacles, starts[0] if starts else None, goals[0] if goals else None, corners
        except _RecognitionError:
            continue


def format_map_result(result: DetectionResult[MapData]) -> str:
    """결과를 한국어 문구로 만듭니다. 실제 출력은 호출부에서 처리합니다."""
    if not result.ok:
        return f"맵 인식 실패: {result.error_code}"
    data = result.data
    start = '없음 (미검출)' if data.start is None else str(data.start)
    goal = '없음 (미검출)' if data.goal is None else str(data.goal)
    return (f"인식 방식: {_last_recognition_method}\n"
            f"맵: {data.rows}행 × {data.cols}열\n"
            f"시작점: {start}\n도착점: {goal}\n"
            f"장애물: {sum(map(sum, data.grid))}개")


'''
사진 한 장을 받아 프로그램이 사용할 수 있는 격자 지도 데이터로 바꾸는 함수

전체 코드에서 왜 필요한가?
main()이 사진 파일을 읽은 직후의 frame에는 픽셀의 색상만 들어 있음. 
격자가 몇 행 몇 열인지, 어느 칸이 장애물인지는 아직 알 수 없음. 
main()에서 이 정보를 얻는 부분이 다음 한 줄.
result = detect_map(frame)
detect_map은 격자를 찾고 칸을 판별하는 여러 보조 함수를 순서대로 호출한 뒤, 그 결과를 DetectionResult 하나로 묶어 돌려줌. 
덕분에 main()은 result.ok로 성공 여부를 확인하고, 성공하면 result.data.grid를 출력하거나 draw_map_preview()로 원본 사진 위에 격자를 그릴 수 있음.
'''
def detect_map(frame: np.ndarray) -> DetectionResult[MapData]:
    """OpenCV BGR 이미지 한 장을 인식하고 결과 또는 오류 코드를 반환합니다.

    행과 열이 각각 2~40개인 격자를 대상으로 합니다.
    화면 기준으로 대략 바르게 놓인 맵을 가정합니다.
    격자만으로는 게임에서 의도한 맵의 회전 방향을 알 수 없습니다.
    공통 모델 생성자의 오류는 통합 과정에서 확인할 수 있도록 그대로 전달합니다.
    """
    # 1. 입력 사진을 검사.
    global _last_recognition_method
    _last_recognition_method = None

    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) < 32):
        return _failure('INVALID_FRAME')
    
    # 2. 인식에 사용할 사진 크기를 정하기
    # 처리량을 줄이기 위해 축소하고, 검출 좌표는 원본 픽셀 좌표로 복원합니다.
    scale = min(1.0, 1200 / max(frame.shape[:2]))
    working = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    coordinate_scale = np.array([ frame.shape[1] / working.shape[1], frame.shape[0] / working.shape[0] ], dtype=np.float32)
    last_error = 'GRID_NOT_FOUND'
    recognized = []
    
    # 3. 먼저 사각형 테두리를 기준으로 격자를 찾기
    method = "일반 맵 인식"
    try:
        # 3.1 _quad_candidates()가 격자의 바깥 테두리처럼 보이는 사각형 후보들을 찾기
        for quad in _quad_candidates(working):
            try:
                # 3.2 각 후보에 대해 _geometry()가 정면에서 본 모양으로 보정하고 가로·세로 격자선을 찾는다.
                warped, xs, ys, corners = _geometry(working, quad)
                # 3.3 이어서 _read_cells()가 각 칸의 안쪽을 잘라 _classify()에 넘긴다. _read_cells함수 안에 label = _classify(cell) 코드 존재
                values = _read_cells(warped, xs, ys)
                # 3.4 끝까지 판독한 후보는 recognized에 저장하고, 한 후보에서 인식 오류가 나면 다음 후보를 검사
                recognized.append((*values, corners * coordinate_scale))
            except _RecognitionError as exc:
                last_error = str(exc)
    except cv2.error:
        return _failure('IMAGE_PROCESSING_FAILED')
    
    # 4. 사각형 방식으로 읽은 후보가 하나도 없을 때만 손그림 방식을 시도.
    if not recognized:
        method = "손그림 맵 인식"
        try:
            for directional in (False, True):
                # _handdrawn_maps(): 사진 속 펜 선분을 가로선과 세로선으로 묶고, 그 교점을 이용해 각 칸의 위치를 구하기
                # 먼저 일반 방식(directional=False)으로 시도하고, 결과가 없으면 가로획과 세로획을 따로 추출하는 방식(directional=True)으로 다시 시도
                for values in _handdrawn_maps(working, directional):
                    recognized.append((*values[:-1], values[-1] * coordinate_scale))
                if recognized:
                    break
        except cv2.error:
            return _failure('IMAGE_PROCESSING_FAILED')
    
    # 후보들을 확인하고 최종 결과를 반환. 후보가 없으면 저장해 둔 last_error를 반환. 기본값은 GRID_NOT_FOUND입니다
    if not recognized:
        return _failure(last_error)
    
    # 같은 외곽의 중복 검출은 허용하되 결과가 서로 충돌하면 실패 처리합니다.
    # 후보가 여러 개면 첫 후보와 나머지 후보의 행·열 수, 장애물 배치, 출발점·도착점이 일치하는지 확인. 
    # 네 꼭짓점 중 하나라도 위치 차이가 원본 사진 긴 쪽 길이의 4%를 넘거나, 앞의 지도 정보가 다르면 MULTIPLE_MAPS_OR_AMBIGUOUS_GRID를 반환.
    first = recognized[0]
    for other in recognized[1:]:
        if (first[:2] != other[:2] or first[3:5] != other[3:5]
                or not np.array_equal(first[2], other[2])
                or np.max(np.linalg.norm(first[5] - other[5], axis=1))
                > max(frame.shape[:2]) * 0.04):
            return _failure('MULTIPLE_MAPS_OR_AMBIGUOUS_GRID')
    _last_recognition_method = method
    
    # 후보가 하나이거나 여러 후보가 일치하면 _success(*first)로 지도 데이터를 반환
    # 여기에는 행·열 수, 장애물 배열(0은 이동 가능, 1은 장애물), 출발점·도착점의 (행, 열), 원본 사진에서의 네 꼭짓점 좌표가 들어간다. 
    return _success(*first)


def get_cell_centers(data: MapData) -> np.ndarray:
    """cell_centers[행, 열]에 대응하는 원본 사진의 중심 픽셀 (x, y)를 반환합니다."""
    rows, cols = data.rows, data.cols
    bounds = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    transform = cv2.getPerspectiveTransform(bounds, np.asarray(data.corners, np.float32))
    yy, xx = np.indices((rows, cols), dtype=np.float32)
    centers = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    # 원본 사진 좌표의 소수부는 유지하고, 화면에 그릴 때만 반올림합니다.
    return cv2.perspectiveTransform(centers, transform)


def draw_map_preview(frame: np.ndarray, data: MapData) -> np.ndarray:
    """인식 결과를 원본 사진의 복사본 위에 그립니다."""
    rows, cols = data.rows, data.cols
    bounds = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    transform = cv2.getPerspectiveTransform(bounds, np.asarray(data.corners, np.float32))
    # 논리 격자의 교점을 사진 좌표로 옮깁니다. 손그림 내부 선 위치는 근사값입니다.
    yy, xx = np.indices((rows + 1, cols + 1), dtype=np.float32)
    vertices = cv2.perspectiveTransform(np.stack((xx, yy), axis=-1), transform)
    vertices = np.rint(vertices).astype(np.int32)
    cell_centers = get_cell_centers(data)
    preview = frame.copy()
    thickness = max(1, round(min(frame.shape[:2]) / 500))
    for row in range(rows):
        for col in range(cols):
            if data.grid[row][col] == 1:
                polygon = np.array([vertices[row, col], vertices[row, col+1],
                                    vertices[row+1, col+1], vertices[row+1, col]])
                cv2.fillConvexPoly(preview, polygon, (0, 155, 255), cv2.LINE_AA)
    # 장애물 칸 아래의 실제 필기도 보이도록 색을 반투명하게 합성합니다.
    cv2.addWeighted(preview, 0.55, frame, 0.45, 0, dst=preview)
    for line in [*vertices, *vertices.transpose(1, 0, 2)]:
        cv2.polylines(preview, [np.ascontiguousarray(line)], False,
                      (60, 190, 70), thickness, cv2.LINE_AA)
    for position, label, color in ((data.start, 'S', (35, 145, 35)),
                                   (data.goal, 'G', (30, 30, 210))):
        if position is None:
            continue
        row, col = position
        corner = vertices[row, col]
        side = min(np.linalg.norm(vertices[row+1, col] - corner),
                   np.linalg.norm(vertices[row, col+1] - corner))
        radius = max(2, round(side * 0.12))
        x, y = map(int, np.rint(cell_centers[row, col]))
        # 출발·도착 표식은 중심점과 주소를 가리지 않도록 위쪽에 둡니다.
        center = (x, y - 2 * radius)
        cv2.circle(preview, center, radius, color, -1, cv2.LINE_AA)
        font_scale = radius / 20
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        origin = (center[0] - width // 2, center[1] + height // 2)
        cv2.putText(preview, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    for row in range(rows):
        for col in range(cols):
            x, y = map(int, np.rint(cell_centers[row, col]))
            side = min(np.linalg.norm(vertices[row+1, col] - vertices[row, col]),
                       np.linalg.norm(vertices[row, col+1] - vertices[row, col]))
            radius = max(1, min(4, round(side * 0.025)))
            # 밝은 테두리 덕분에 검은 장애물 위에서도 중심점이 보입니다.
            cv2.circle(preview, (x, y), radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(preview, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)
            px, py = cell_centers[row, col]
            labels = (f"({row},{col})", f"[{px:.2f}, {py:.2f}]")
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_width = max(cv2.getTextSize(label, font, 1, 1)[0][0] for label in labels)
            scale = min(0.7, side * 0.82 / base_width)
            baseline_y = y + radius + 2
            for label in labels:
                (width, height), descent = cv2.getTextSize(label, font, scale, 1)
                baseline_y += height
                origin = (x - width // 2, baseline_y)
                cv2.putText(preview, label, origin, font, scale, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(preview, label, origin, font, scale, (0, 0, 0), 1, cv2.LINE_AA)
                baseline_y += descent + 1
    return preview


def main() -> int:
    # modules/map_recognition.py를 기준으로 프로젝트의 사진을 찾기.
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "test_pictures" / "picture_11.png"

    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error) as exc:
        print(f"사진을 읽을 수 없습니다: {image_path}\n{exc}")
        return 2

    if frame is None:
        print(f"이미지 형식을 확인해 주세요: {image_path}")
        return 2

    result = detect_map(frame)

    print(format_map_result(result))
    if not result.ok:
        return 1

    print("장애물 배치 (0: 이동 가능, 1: 장애물):")
    for row in result.data.grid:
        print(row)
    print("격자 꼭짓점 (왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래):")
    print(result.data.corners)
    preview = draw_map_preview(frame, result.data)
    title = "CardRoad | Map preview"
    print("초록 선: 격자 / 주황색: 장애물 / 초록 S: 출발점 / 빨간 G: 도착점")
    print("미리보기 창에서 아무 키나 누르거나 창을 닫으면 종료합니다.")
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        height, width = preview.shape[:2]
        scale = min(1.0, 1100 / width, 750 / height)
        cv2.resizeWindow(title, round(width * scale), round(height * scale))
        cv2.imshow(title, preview)
        while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(50) >= 0:
                break
    except cv2.error as exc:
        print(f"미리보기 창을 표시할 수 없습니다: {exc}")
        return 2
    else:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())