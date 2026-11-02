"""사진 데모용 격자 칸 수 추정. 팀의 맵 인식 모듈/공통 인터페이스는 수정하지 않습니다.

밝은 바탕의 어두운 직선, 균일한 격자 간격을 가정합니다.
네 꼭짓점은 종이 외곽이 아닌 실제 격자 외곽이어야 합니다.
"""
import cv2
import numpy as np


class GridSizeError(ValueError):
    """칸 수를 신뢰할 만큼 격자선을 확인하지 못했습니다."""


def _count_cells(mask, axis):
    # 격자를 충분히 가로지르는 선만 선택해 글자나 작은 도형을 제외합니다.
    coverage = np.mean(mask > 0, axis=axis)
    indices = np.flatnonzero(coverage >= .60)
    if not indices.size:
        raise GridSizeError("격자선을 찾지 못했습니다. 격자 전체가 선명하게 보이는 사진을 선택해 주세요.")
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
    length = len(coverage) - 1
    centers = []
    for group in groups:
        center = float(np.average(group, weights=coverage[group]))
        # 외곽선은 클릭으로 제공됐으므로 칸 사이의 내부 선만 셉니다.
        if center < length * .012 or center > length * .988:
            continue
        if group[-1] - group[0] > length * .025:
            raise GridSizeError("격자선과 넓은 검은 영역을 구분하기 어렵습니다. 더 선명한 사진이 필요합니다.")
        centers.append(center)
    cells = len(centers) + 1
    if not 2 <= cells <= 40:
        raise GridSizeError("이 예제는 가로·세로 2~40칸의 격자를 지원합니다. 내부 격자선이 보이는지 확인해 주세요.")
    expected = np.arange(1, cells) * length / cells
    tolerance = max(5., length / cells * .10)
    if np.max(np.abs(np.asarray(centers) - expected)) > tolerance:
        raise GridSizeError("격자선 간격이 일정하지 않거나 일부 선이 가려졌습니다. 격자 외곽 네 점을 다시 지정해 주세요.")
    return cells


def detect_grid_size(frame, corners):
    """원근 보정 후 (rows, cols)를 반환합니다. 불확실하면 GridSizeError를 냅니다."""
    points = np.asarray(corners, dtype=np.float32)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 100):
        raise GridSizeError("격자 외곽의 네 꼭짓점을 순서대로 지정해 주세요.")
    size = 1000
    target = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    transform = cv2.getPerspectiveTransform(points, target)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flat = cv2.warpPerspective(gray, transform, (size, size), borderValue=255)
    binary = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 9)
    # 작은 끊김을 연결한 뒤 긴 가로/세로 선만 남깁니다.
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1, 300), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((300, 1), np.uint8))
    return _count_cells(horizontal, axis=1), _count_cells(vertical, axis=0)
