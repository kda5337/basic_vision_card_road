"""독립 AR 테스트: python -m examples.demo_ar (카메라/인식 모듈 불필요)."""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from common.models import (CardData, CharacterState, Command, Direction,
                           ExecutionResult, ExecutionStep, GameStatus, MapData)
from modules.ar_visualization import ARVisualizer
from examples.grid_size import detect_grid_size


SCENARIOS = ("success", "collision", "bounds", "syntax", "repeat", "turns")
LAYOUTS = {
    "topview": [[285, 100], [675, 100], [675, 490], [285, 490]],
    "perspective": [[150, 100], [760, 145], [830, 510], [90, 480]],
    "rectangle": [[130, 100], [800, 100], [800, 490], [130, 490]],
    "steep": [[290, 150], [620, 100], [850, 470], [100, 490]],
}


def topview_corners(rows, cols):
    """수직 촬영 기준: 가로·세로 칸 수가 달라도 각 칸은 같은 크기의 정사각형."""
    cell = min(390 / rows, 670 / cols)
    left = (960 - cols * cell) / 2
    return np.float32([[left, 100], [left + cols * cell, 100],
                       [left + cols * cell, 100 + rows * cell], [left, 100 + rows * cell]])


def sample(rows=5, cols=5, corners=None, scenario="success"):
    """통합 전 임시 데이터입니다. 게임 엔진을 대신 구현하지 않습니다."""
    if rows < 2 or cols < 2:
        raise ValueError("Demo requires at least 2 rows and 2 columns")
    if corners is None:
        corners = topview_corners(rows, cols)
    board = MapData(rows, cols, [[0] * cols for _ in range(rows)],
                    (rows - 1, 0), (rows - 2, 1), np.float32(corners))
    states = [CharacterState(rows - 1, 0, Direction.UP),
              CharacterState(rows - 2, 0, Direction.UP),
              CharacterState(rows - 2, 0, Direction.RIGHT),
              CharacterState(rows - 2, 1, Direction.RIGHT)]
    commands = [Command.FORWARD, Command.TURN_RIGHT, Command.FORWARD]
    steps = [ExecutionStep(i, command, states[i], states[i + 1])
             for i, command in enumerate(commands)]
    result = ExecutionResult(GameStatus.SUCCESS, steps, states[-1])
    start = states[0]
    if scenario in ("collision", "bounds"):
        direction = Direction.RIGHT if scenario == "collision" else Direction.LEFT
        turn = Command.TURN_RIGHT if scenario == "collision" else Command.TURN_LEFT
        after = CharacterState(rows - 1, 0, direction)
        attempted = (rows - 1, 1 if scenario == "collision" else -1)
        commands = [turn, Command.FORWARD]
        if scenario == "collision":
            board.grid[attempted[0]][attempted[1]] = 1
        result = ExecutionResult(GameStatus.FAILURE,
                                 [ExecutionStep(0, turn, start, after)], after,
                                 "OBSTACLE_COLLISION" if scenario == "collision" else "OUT_OF_BOUNDS",
                                 1, attempted)
    elif scenario == "syntax":
        commands = [Command.REPEAT_2, Command.FORWARD]
        result = ExecutionResult(GameStatus.ERROR, [], start, "UNCLOSED_LOOP", 0)
    elif scenario == "repeat":
        commands = [Command.REPEAT_2, Command.TURN_LEFT, Command.TURN_RIGHT, Command.LOOP_END]
        left = CharacterState(rows - 1, 0, Direction.LEFT)
        # 반복이 펼쳐진 실행 기록을 직접 지정합니다. AR에서는 반복문을 해석하지 않습니다.
        steps = [ExecutionStep(1, Command.TURN_LEFT, start, left),
                 ExecutionStep(2, Command.TURN_RIGHT, left, start),
                 ExecutionStep(1, Command.TURN_LEFT, start, left),
                 ExecutionStep(2, Command.TURN_RIGHT, left, start)]
        result = ExecutionResult(GameStatus.FAILURE, steps, start, "GOAL_NOT_REACHED")
    elif scenario == "turns":
        commands = [Command.TURN_LEFT, Command.TURN_RIGHT, Command.TURN_RIGHT, Command.FORWARD]
        states = [start, CharacterState(rows - 1, 0, Direction.LEFT), start,
                  CharacterState(rows - 1, 0, Direction.RIGHT),
                  CharacterState(rows - 1, 1, Direction.RIGHT)]
        board.goal = (rows - 1, 1)
        steps = [ExecutionStep(i, cmd, states[i], states[i + 1]) for i, cmd in enumerate(commands)]
        result = ExecutionResult(GameStatus.SUCCESS, steps, states[-1])
    elif scenario != "success":
        raise ValueError(f"Unknown scenario: {scenario}")
    # 35:45 비율, 맵 아래 왼쪽→오른쪽 밀착 배치. 실제 인쇄 디자인은 별도 원본 사용.
    cards = [CardData(command, (16 + i * 56, 514, 56, 72))
             for i, command in enumerate(commands)]
    return board, cards, result


@lru_cache(maxsize=1)
def map_markers():
    """사용자 제공 그림에서 글씨와 외곽 테두리를 제외한 아이콘 영역만 사용합니다."""
    assets = Path(__file__).resolve().parent / "assets"
    sheet = cv2.imdecode(np.fromfile(assets / "map_markers.png", np.uint8), cv2.IMREAD_COLOR)
    if sheet is None:
        raise ValueError("예시 맵의 마커 이미지를 읽을 수 없습니다.")
    # 고정된 첨부 이미지의 내부 영역. 아래 START/GOAL 글씨와 사각 테두리는 표시하지 않습니다.
    return {"start": sheet[22:116, 9:106], "goal": sheet[22:116, 132:228]}


def draw_map_marker(frame, matrix, row, col, marker):
    """고정 마커를 원래 비율로 칸 안에 넣고 맵의 원근에 맞춰 표시합니다."""
    height, width = marker.shape[:2]
    scale = .84 / max(width, height)
    marker_width, marker_height = width * scale, height * scale
    left, top = col + (1 - marker_width) / 2, row + (1 - marker_height) / 2
    target = np.float32([[[left, top], [left + marker_width, top],
                          [left + marker_width, top + marker_height], [left, top + marker_height]]])
    pixels = cv2.perspectiveTransform(target, matrix)[0]
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [width - 1, 0],
                                                        [width - 1, height - 1], [0, height - 1]]), pixels)
    size = (frame.shape[1], frame.shape[0])
    texture = cv2.warpPerspective(marker, transform, size, borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform, size,
                               flags=cv2.INTER_NEAREST).astype(bool)
    frame[mask] = texture[mask]


def synthetic_frame(board, cards=()):
    frame = np.full((600, 960, 3), (38, 32, 28), dtype=np.uint8)
    src = np.float32([[0, 0], [board.cols, 0],
                     [board.cols, board.rows], [0, board.rows]])
    matrix = cv2.getPerspectiveTransform(src, board.corners)
    markers = map_markers()
    for row in range(board.rows):
        for col in range(board.cols):
            cell = np.float32([[[col, row], [col + 1, row],
                                [col + 1, row + 1], [col, row + 1]]])
            polygon = np.rint(cv2.perspectiveTransform(cell, matrix)[0]).astype(np.int32)
            # 장애물은 별도 라벨/테두리 없이 칸 전체를 검게 채웁니다.
            obstacle = board.grid[row][col] and (row, col) not in (board.start, board.goal)
            cv2.fillConvexPoly(frame, polygon, (20, 20, 20) if obstacle else (255, 255, 255))
            cv2.polylines(frame, [polygon], True, (45, 45, 45), 2, cv2.LINE_AA)
            kind = ("start" if (row, col) == board.start else
                    "goal" if (row, col) == board.goal else None)
            if kind:
                draw_map_marker(frame, matrix, row, col, markers[kind])
    for card in cards:
        if card.bbox is None:
            continue
        x, y, width, height = card.bbox
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), -1)
        cv2.rectangle(frame, (x + 2, y + 2), (x + width - 3, y + height - 3), (0, 0, 0), 2)
        labels = {Command.FORWARD: "FWD", Command.TURN_LEFT: "LEFT", Command.TURN_RIGHT: "RIGHT",
                  Command.REPEAT_2: "x2", Command.REPEAT_3: "x3", Command.LOOP_END: "END"}
        cv2.putText(frame, labels[card.command], (x + 5, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def fit_photo(frame, corners, max_width, max_height):
    """큰 사진을 화면에 맞추고 지정 좌표도 같은 비율로 변환합니다."""
    height, width = frame.shape[:2]
    scale = min(1., max_width / width, max_height / height)
    target_width, target_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (target_width, target_height) == (width, height):
        return frame, corners
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if corners is not None:
        corners = np.asarray(corners, np.float32) * [target_width / width, target_height / height]
    return resized, corners


def select_corners(frame, window):
    """선택한 좌표는 표시 영상과 같은 픽셀 좌표계입니다."""
    points = []

    def pick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, pick)
    while True:
        preview = frame.copy()
        valid = (len(points) == 4 and cv2.isContourConvex(np.float32(points))
                 and abs(cv2.contourArea(np.float32(points))) >= 1.)
        text = "Click grid: 1-TL 2-TR 3-BR 4-BL | R: reset | Esc: quit"
        if len(points) == 4:
            text = "Enter: confirm | R: reset" if valid else "Invalid corners. R: reset"
        # 안내문 영역을 어둡게 해 밝은 종이 사진에서도 글씨가 보이게 합니다.
        cv2.rectangle(preview, (0, 0), (min(frame.shape[1] - 1, 850), 42), (30, 30, 30), -1)
        cv2.putText(preview, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.polylines(preview, [np.int32(points)], len(points) == 4, (0, 220, 255), 2)
        for i, point in enumerate(points):
            cv2.circle(preview, point, 5, (0, 0, 255), -1)
            cv2.putText(preview, str(i + 1), (point[0] + 8, point[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        if key in (10, 13) and valid:
            return np.float32(points)
        if key in (ord("r"), ord("R")):
            points.clear()
        elif key == 8 and points:
            points.pop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="촬영 이미지; 격자 외곽 꼭짓점 4개를 클릭")
    parser.add_argument("--auto-grid", action="store_true", help="사진의 행·열 개수 자동 인식")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--scenario", choices=SCENARIOS, default="success")
    parser.add_argument("--layout", choices=LAYOUTS, default="topview")
    parser.add_argument("--corners", type=float, nargs=8, metavar="N",
                        help="TL TR BR BL 순서의 x y 좌표 8개; --image와 함께 사용 가능")
    parser.add_argument("--output", type=Path, help="GUI 없이 예제 첫 프레임을 저장")
    parser.add_argument("--max-width", type=int, default=1200, help="사진 표시 최대 너비")
    parser.add_argument("--max-height", type=int, default=750, help="사진 표시 최대 높이")
    args = parser.parse_args(argv)
    if args.auto_grid and not args.image:
        parser.error("--auto-grid requires --image")
    if args.max_width <= 0 or args.max_height <= 0:
        parser.error("Display dimensions must be positive")
    if args.rows < 2 or args.cols < 2:
        parser.error("Demo requires at least 2 rows and 2 columns")
    if args.image and args.output and args.corners is None:
        parser.error("Use --corners with --image and --output")
    corners = (np.float32(args.corners).reshape(4, 2) if args.corners is not None
               else (topview_corners(args.rows, args.cols) if args.layout == "topview"
                     else np.float32(LAYOUTS[args.layout])))
    scenario = args.scenario
    board, cards, result = sample(args.rows, args.cols, corners, scenario)
    window = "CardRoad AR demo"
    if args.image:
        # Windows의 한글 파일 경로도 읽을 수 있습니다.
        frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read image: {args.image}")
        if not args.output:
            frame, corners = fit_photo(frame, corners, args.max_width, args.max_height)
        if args.corners is None:
            corners = select_corners(frame, window)
            if corners is None:
                return
        if args.auto_grid:
            try:
                args.rows, args.cols = detect_grid_size(frame, corners)
            except ValueError:
                cv2.destroyAllWindows()
                raise
            print(f"Detected grid: {args.rows} rows x {args.cols} columns")
        board, cards, result = sample(args.rows, args.cols, corners, scenario)
        # 사진에서 실제 카드 위치를 인식하지 않았으므로 가짜 박스를 덧그리지 않습니다.
        for card in cards:
            card.bbox = None
    else:
        frame = synthetic_frame(board, cards)
    ar = ARVisualizer()
    if args.output:
        ar.begin(board, cards, result)
        output = ar.render(frame, f"AR 단독 검사 · 세로 {args.rows}칸 × 가로 {args.cols}칸 · 테스트용 이동 기록")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", output)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        encoded.tofile(args.output)
        print(args.output.resolve())
        return
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and ar.is_start_clicked(x, y, frame.shape):
            ar.begin(board, cards, result)

    cv2.setMouseCallback(window, click)
    try:
        while True:
            help_text = (f"AR 단독 검사 · {args.rows}행 × {args.cols}열 · "
                         f"{scenario} · Space: 재생 / 1~6: 예제 / Esc: 종료")
            cv2.imshow(window, ar.render(frame, help_text))
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" ") and ar.is_finished():
                ar.begin(board, cards, result)
            elif ord("1") <= key <= ord("6"):
                scenario = SCENARIOS[key - ord("1")]
                board, cards, result = sample(args.rows, args.cols, corners, scenario)
                if args.image:
                    for card in cards:
                        card.bbox = None
                else:
                    frame = synthetic_frame(board, cards)
                ar = ARVisualizer()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
