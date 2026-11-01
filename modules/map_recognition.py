import cv2
import numpy as np
import argparse

from common.models import MapData, DetectionResult
from pathlib import Path


WARP_SIZE = 720
MAX_CELLS = 40


class _RecognitionError(Exception):
    pass


# 인식 실패 원인을 오류 코드로 반환.
def _failure(code):
    return DetectionResult(data=None, error_code=code)


# 격자 크기와 칸별 정보를 맵 인식 결과로 묶어 반환.
def _success(rows, cols, obstacles, start, goal, corners):
    return DetectionResult(
        data=MapData(
            rows=rows, cols=cols, grid=obstacles.astype(np.uint8).tolist(),
            start=start, goal=goal, corners=np.asarray(corners, dtype=np.float32),
        ),
        error_code=None,
    )


# 네 꼭짓점을 왼쪽 위부터 시계 방향으로 정렬.
def _order_quad(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0).copy()


# 사진에서 격자 외곽이 될 만한 사각형 후보를 찾기.
def _quad_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9,)
    edges = cv2.Canny(gray, 40, 130)


    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


    min_area = frame.shape[0] * frame.shape[1] * 0.025
    candidates = []
    for mask in (binary, edges):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours:

            if abs(cv2.contourArea(contour)) < min_area:
                continue


            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.015, 0.025, 0.04):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):

                    continue


                quad = _order_quad(polygon)
                if np.min(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) < 35:
                    continue
                if not any(np.mean(np.linalg.norm(quad - old, axis=1)) < 8
                           for old in candidates):
                    candidates.append(quad)
                break


    return sorted(candidates, key=cv2.contourArea, reverse=True)[:30]


# 가로 또는 세로 격자선의 위치를 찾고 간격을 검사.
def _line_positions(mask, vertical):
    length = max(15, mask.shape[0 if vertical else 1] // 6)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, length) if vertical else (length, 1)
    )
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    profile = (lines > 0).mean(axis=0 if vertical else 1)
    indices = np.flatnonzero(profile >= 0.40)
    if len(indices) == 0:
        raise _RecognitionError('GRID_NOT_FOUND')
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)

    positions = np.array([
        np.average(group, weights=profile[group])
        for group in groups if len(group) <= WARP_SIZE * 0.035
    ], dtype=np.float32)
    if not 3 <= len(positions) <= MAX_CELLS + 1:
        raise _RecognitionError('GRID_NOT_FOUND')
    gaps = np.diff(positions)
    spacing = np.median(gaps)

    if spacing < 12 or np.max(np.abs(gaps / spacing - 1)) > 0.22:
        raise _RecognitionError('GRID_IRREGULAR')
    if positions[-1] - positions[0] < WARP_SIZE * 0.5:
        raise _RecognitionError('GRID_NOT_FOUND')
    return positions


# 사각형 후보를 정면으로 보정해 격자선과 바깥 꼭짓점을 구하기.
def _geometry(frame, quad):
    n = WARP_SIZE

    destination = np.array([[12, 12], [n-13, 12], [n-13, n-13], [12, n-13]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(frame, matrix, (n, n), borderValue=(255, 255, 255))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9,
    )
    xs = _line_positions(mask, True)
    ys = _line_positions(mask, False)
    outer = np.array([
        [xs[0], ys[0]], [xs[-1], ys[0]],
        [xs[-1], ys[-1]], [xs[0], ys[-1]],
    ], np.float32)
    corners = cv2.perspectiveTransform(outer[None], np.linalg.inv(matrix))[0]
    return warped, xs, ys, corners


# 칸 안쪽의 색과 펜 자국으로 출발점, 도착점, 장애물, 빈칸을 구분.
def _classify(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    blue_ch, green_ch, red_ch = cv2.split(cell.astype(np.int16))
    green = ((hue >= 35) & (hue <= 85) & (saturation >= 45) & (value >= 35)
             & (green_ch - np.maximum(blue_ch, red_ch) >= 10))
    red = ((hue <= 12) | (hue >= 168)) & (saturation >= 85) & (value >= 65)
    green_ratio, red_ratio = green.mean(), red.mean()
    dark_ratio = ((value < 85) & ~green & ~red).mean()

    has_green = green_ratio >= 0.03
    if has_green and red_ratio >= 0.045:
        raise _RecognitionError('CELL_AMBIGUOUS')
    if has_green:
        return 'start'
    if red_ratio >= 0.045:
        return 'goal'


    small_v = cv2.resize(value, (64, 64), interpolation=cv2.INTER_AREA)
    small_s = cv2.resize(saturation, (64, 64), interpolation=cv2.INTER_AREA)
    paper = cv2.morphologyEx(small_v, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    marker_fraction = cv2.resize((green | red).astype(np.float32), (64, 64),
                                 interpolation=cv2.INTER_AREA)
    stroke_ratio = ((cv2.subtract(paper, small_v) > 32)
                    & (small_s < 115) & (marker_fraction < 0.5)).mean()
    if dark_ratio >= 0.55 or stroke_ratio >= 0.22:
        return 'obstacle'

    unexpected_color = ((saturation > 100) & (value > 85)).mean()
    if dark_ratio > 0.15 or unexpected_color > 0.25:
        raise _RecognitionError('CELL_AMBIGUOUS')
    return 'empty'


# 각 칸을 판별해 장애물 배치와 출발점·도착점 좌표를 구하기.
def _read_cells(warped, xs, ys):
    rows, cols = len(ys) - 1, len(xs) - 1
    obstacles = np.zeros((rows, cols), dtype=bool)
    starts, goals = [], []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = xs[col:col+2]
            y0, y1 = ys[row:row+2]
            mx, my = (x1-x0)*0.17, (y1-y0)*0.17
            cell = warped[
                int(np.ceil(y0+my)):int(np.floor(y1-my)),
                int(np.ceil(x0+mx)):int(np.floor(x1-mx)),
            ]
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


# 비슷한 방향과 위치의 선분을 묶어 손그림 격자선을 추정.
def _stroke_lines(mask, vertical):

    h, w = mask.shape
    span = h if vertical else w
    extent = w if vertical else h
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 360, threshold=max(20, int(span * .10)),
        minLineLength=max(25, int(span * .22)), maxLineGap=max(8, int(span * .08)),
    )
    if segments is None or not segments.size:
        raise _RecognitionError('GRID_NOT_FOUND')
    segments = np.asarray(segments).reshape(-1, 4)

    records = []
    for x1, y1, x2, y2 in segments:
        u1, v1, u2, v2 = (y1, x1, y2, x2) if vertical else (x1, y1, x2, y2)
        if abs(u2-u1) < 1 or abs(v2-v1) > abs(u2-u1) * .35:
            continue
        slope = (v2-v1) / (u2-u1)
        offset = v1 - slope*u1
        records.append((slope*span/2 + offset, u1, v1, u2, v2))
    records.sort()
    groups = []
    for record in records:
        if not groups or record[0] - np.median([r[0] for r in groups[-1]]) > max(5, extent*.035):
            groups.append([])
        groups[-1].append(record)
    fitted = []
    for group in groups:
        u = np.array([v for r in group for v in (r[1], r[3])], float)
        v = np.array([v for r in group for v in (r[2], r[4])], float)
        if np.ptp(u) < span*.55:
            continue
        slope, offset = np.polyfit(u, v, 1)
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


# 가로획과 세로획을 따로 추출해 격자선과 대각선 빗금을 구분.
def _directional_strokes(ink):

    masks = []
    for vertical in (True, False):
        strokes = np.zeros_like(ink)
        for tilt in range(-5, 6):
            kernel = np.zeros((33, 33), np.uint8)
            a, b = (16-tilt, 0), (16+tilt, 32)
            if not vertical:
                a, b = a[::-1], b[::-1]
            cv2.line(kernel, a, b, 1, 1)
            strokes |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        masks.append(strokes)
    return masks


# 손그림의 선과 교점을 찾아 격자를 구성하고 각 칸의 내용을 판별.
def _handdrawn_maps(frame, directional=False):


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 9)
    if directional:
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        ink = ((background.astype(np.float32)-gray > 25) & (gray < 166)
               & (saturation < 125)).astype(np.uint8)*255
        vertical_mask, horizontal_mask = _directional_strokes(ink)
        ink = vertical_mask | horizontal_mask
        gap = 15
    else:
        ink = ((background.astype(np.float32)-gray > 40)
               & (gray < 135) & (background > 140)).astype(np.uint8)*255
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        vertical_mask = horizontal_mask = ink
        gap = 13
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
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
                    px = (ax*by+bx)/(1-ax*ay)
                    py = ay*px+by
                    if not (-w*.1 <= px <= w*1.1 and -h*.1 <= py <= h*1.1):
                        raise _RecognitionError('GRID_IRREGULAR')
                    points[r, c] = (px+x, py+y)
            rows, cols = len(horizontal)-1, len(vertical)-1
            obstacles = np.zeros((rows, cols), bool)
            starts, goals = [], []
            destination = np.float32([[0, 0], [63, 0], [63, 63], [0, 63]])
            for r in range(rows):
                for c in range(cols):
                    cell_quad = np.array([points[r,c], points[r,c+1],
                                          points[r+1,c+1], points[r+1,c]], np.float32)

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


# 맵 인식 결과를 화면에 출력할 한국어 문구로 만들기.
def format_map_result(result: DetectionResult[MapData]) -> str:

    if not result.ok:
        return f"맵 인식 실패: {result.error_code}"
    data = result.data
    start = '없음 (미검출)' if data.start is None else str(data.start)
    goal = '없음 (미검출)' if data.goal is None else str(data.goal)
    return (f"맵: {data.rows}행 × {data.cols}열\n"
            f"시작점: {start}\n도착점: {goal}\n"
            f"장애물: {sum(map(sum, data.grid))}개")


# 입력 사진에서 격자를 찾아 맵 정보 또는 오류 코드를 반환.
def detect_map(frame: np.ndarray) -> DetectionResult[MapData]:


    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) < 32):
        return _failure('INVALID_FRAME')

    scale = min(1.0, 1200 / max(frame.shape[:2]))
    working = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    coordinate_scale = np.array([
        frame.shape[1] / working.shape[1], frame.shape[0] / working.shape[0]
    ], dtype=np.float32)
    last_error = 'GRID_NOT_FOUND'
    recognized = []
    try:
        for quad in _quad_candidates(working):
            try:
                warped, xs, ys, corners = _geometry(working, quad)
                values = _read_cells(warped, xs, ys)
                recognized.append((*values, corners * coordinate_scale))
            except _RecognitionError as exc:
                last_error = str(exc)
    except cv2.error:
        return _failure('IMAGE_PROCESSING_FAILED')
    if not recognized:
        try:
            for directional in (False, True):
                for values in _handdrawn_maps(working, directional):
                    recognized.append((*values[:-1], values[-1] * coordinate_scale))
                if recognized:
                    break
        except cv2.error:
            return _failure('IMAGE_PROCESSING_FAILED')
    if not recognized:
        return _failure(last_error)

    first = recognized[0]
    for other in recognized[1:]:
        if (first[:2] != other[:2] or first[3:5] != other[3:5]
                or not np.array_equal(first[2], other[2])
                or np.max(np.linalg.norm(first[5] - other[5], axis=1))
                > max(frame.shape[:2]) * 0.04):
            return _failure('MULTIPLE_MAPS_OR_AMBIGUOUS_GRID')
    return _success(*first)


# 각 칸의 중심을 원본 사진의 픽셀 좌표로 계산.
def get_cell_centers(data: MapData) -> np.ndarray:

    rows, cols = data.rows, data.cols
    bounds = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    transform = cv2.getPerspectiveTransform(bounds, np.asarray(data.corners, np.float32))
    yy, xx = np.indices((rows, cols), dtype=np.float32)
    centers = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    return cv2.perspectiveTransform(centers, transform)


# 사진 복사본에 격자, 표식, 장애물과 각 칸의 중심 좌표를 그리기.
def draw_map_preview(frame: np.ndarray, data: MapData) -> np.ndarray:

    rows, cols = data.rows, data.cols
    bounds = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    transform = cv2.getPerspectiveTransform(bounds, np.asarray(data.corners, np.float32))

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


# 사진 파일을 읽어 맵을 인식하고 결과와 미리보기를 표시.
def main() -> int:
    # modules/map_recognition.py를 기준으로 프로젝트의 사진을 찾기.
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "test_pictures" / "picture_8.jpg"

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