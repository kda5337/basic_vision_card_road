"""CardRoad 웹캠 게임 실행 및 시스템 통합 모듈 (담당: 강호현).

실행: python main.py --camera 0
S/Space/시작 버튼: 게임 실행 | M: 맵 확인 | C: 카드 확인 | R: 실시간 화면 | Q/Esc: 종료

맵·카드 검출에는 AR/UI 합성이 없는 동일한 원본 프레임을 전달합니다.
"""

import argparse
import traceback

import cv2
import numpy as np

from common.models import CardData, Command, MapData
from modules.ar_visualization import ARVisualizer
from modules.card_recognition import detect_cards
from modules.command_processor import execute
from modules.map_recognition import detect_map, draw_map_preview, format_map_result


_CARD_LABELS = {
    Command.FORWARD: "F",
    Command.TURN_LEFT: "L",
    Command.TURN_RIGHT: "R",
    Command.REPEAT_2: "x2",
    Command.REPEAT_3: "x3",
    Command.LOOP_END: "END",
}


def draw_card_diagnostics(frame: np.ndarray, cards: list[CardData]) -> np.ndarray:
    """촬영 당시 이미지 위에 카드 순서·종류·신뢰도를 표시합니다."""
    output = frame.copy()
    height, width = output.shape[:2]
    for index, card in enumerate(cards, start=1):
        if card.bbox is None:
            continue
        x, y, w, h = map(int, card.bbox)
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), (45, 210, 45), 3)

        short_name = _CARD_LABELS.get(card.command, str(card.command))
        label = f"{index}:{short_name}"
        if card.confidence is not None:
            label += f" {card.confidence:.2f}"
        size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        text_x = max(0, min(x, width - size[0] - 6))
        text_y = y - 7 if y >= size[1] + 11 else min(height - 4, y + h + size[1] + 5)
        cv2.rectangle(
            output, (text_x, text_y - size[1] - 3),
            (text_x + size[0] + 5, text_y + baseline + 2), (20, 20, 20), -1
        )
        cv2.putText(output, label, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def print_card_diagnostics(cards: list[CardData]) -> None:
    """검출된 카드 전체를 왼쪽부터 오른쪽 순서로 콘솔에 출력합니다."""
    print(f"[카드 인식] 총 {len(cards)}장 (왼쪽 → 오른쪽)")
    for index, card in enumerate(cards, start=1):
        confidence = "미제공" if card.confidence is None else f"{card.confidence:.3f}"
        command_name = card.command.value if isinstance(card.command, Command) else str(card.command)
        print(f"  {index:02d}. {command_name:12s} | 신뢰도={confidence} | bbox={card.bbox}")


def print_map_diagnostics(map_data: MapData) -> None:
    """맵의 크기, 표식, 장애물과 셀별 분류를 콘솔에 출력합니다."""
    obstacles = sum(cell == 1 for row in map_data.grid for cell in row)
    print(f"[맵 인식] {map_data.rows}행 × {map_data.cols}열 "
          f"| 시작={map_data.start} | 도착={map_data.goal} | 장애물={obstacles}개")
    print("  S: 출발, G: 도착, #: 장애물, .: 빈 칸")
    for row in range(map_data.rows):
        symbols = []
        for col in range(map_data.cols):
            if (row, col) == map_data.start:
                symbols.append("S")
            elif (row, col) == map_data.goal:
                symbols.append("G")
            elif row < len(map_data.grid) and col < len(map_data.grid[row]) \
                    and map_data.grid[row][col] == 1:
                symbols.append("#")
            else:
                symbols.append(".")
        print("  " + " ".join(symbols))


def process_game(frame: np.ndarray, ar: ARVisualizer) -> str | None:
    """Start 시점의 원본 프레임으로 인식·실행한 뒤 AR 재생을 시작합니다.

    성공적으로 AR에 실행 기록을 전달하면 None을 반환하며,
    인식 실패 시에는 AR 상태창에 표시할 오류 메시지를 반환합니다.
    """
    map_result = detect_map(frame)
    if not map_result.ok or map_result.data is None:
        return f"맵 인식 실패: {map_result.error_code}"

    map_data: MapData = map_result.data
    print_map_diagnostics(map_data)

    # 격자만 검출되고 필수 색상 마커가 없는 경우에도 실패를 표시합니다.
    if map_data.start is None:
        return "맵 인식 실패: START_NOT_FOUND"
    if map_data.goal is None:
        return "맵 인식 실패: GOAL_NOT_FOUND"

    card_result = detect_cards(frame)
    if not card_result.ok or card_result.data is None:
        return f"카드 인식 실패: {card_result.error_code}"

    cards: list[CardData] = card_result.data
    if not cards:
        return "카드 인식 실패: NO_CARDS"
    print_card_diagnostics(cards)

    result = execute(map_data, [card.command for card in cards])
    print(f"[게임 결과] {result.status.value} | 실행 단계={len(result.steps)}개 "
          f"| 오류={result.error_code} | 오류 카드={result.error_card_index} "
          f"| 시도 좌표={result.attempted_cell}")
    ar.begin(map_data, cards, result)
    return None


def main(argv: list[str] | None = None) -> int:
    """웹캠·키보드·마우스 이벤트를 관리하고 프레임마다 AR을 렌더링합니다."""
    parser = argparse.ArgumentParser(description="CardRoad 웹캠 게임")
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (기본 0)")
    args = parser.parse_args(argv)

    try:
        ar = ARVisualizer()
    except RuntimeError as exc:
        print(f"AR 초기화 실패: {exc}")
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"웹캠을 열 수 없습니다: camera={args.camera}")
        cap.release()
        return 1

    window = "CardRoad"
    pending_start = False
    frame_shape: tuple[int, ...] | None = None
    message: str | None = None
    diagnostic_frame: np.ndarray | None = None

    print("[CardRoad] S/Space/시작 버튼: 게임 | M: 맵 확인 | "
          "C: 카드 확인 | R: 실시간 복귀 | Q/Esc: 종료")

    def on_mouse(event, x, y, flags, param):
        nonlocal pending_start
        if (event == cv2.EVENT_LBUTTONDOWN and frame_shape is not None
                and ar.is_finished() and ar.is_start_clicked(x, y, frame_shape)):
            pending_start = True

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("웹캠 프레임을 읽을 수 없습니다.")
                return 1

            frame_shape = frame.shape
            if pending_start and ar.is_finished():
                pending_start = False
                diagnostic_frame = None
                try:
                    # UI가 합성되지 않은 원본 프레임 복사본만 인식에 전달합니다.
                    message = process_game(frame.copy(), ar)
                    if message is not None:
                        print(message)
                except Exception as exc:
                    traceback.print_exc()
                    message = f"실행 오류: {type(exc).__name__}"

            display_frame = diagnostic_frame if diagnostic_frame is not None else frame
            cv2.imshow(window, ar.render(display_frame, message))
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key in (ord("s"), ord(" ")) and ar.is_finished():
                pending_start = True
            elif key == ord("c") and ar.is_finished():
                # 격자를 못 찾는 상황에서도 카드를 따로 검증할 수 있습니다.
                diagnostic_frame = frame.copy()
                try:
                    card_result = detect_cards(frame.copy())
                    if card_result.ok and card_result.data:
                        print_card_diagnostics(card_result.data)
                        diagnostic_frame = draw_card_diagnostics(
                            diagnostic_frame, card_result.data)
                        message = (f"카드 {len(card_result.data)}장 인식 완료 "
                                   "· R: 실시간 화면 · S: 게임 시작")
                    else:
                        error = card_result.error_code or "NO_CARDS"
                        message = f"카드 인식 실패: {error} · R: 실시간 화면"
                        print(message)
                except Exception as exc:
                    traceback.print_exc()
                    message = f"카드 인식 실행 오류: {type(exc).__name__}"
            elif key == ord("m") and ar.is_finished():
                diagnostic_frame = frame.copy()
                try:
                    map_result = detect_map(frame.copy())
                    if map_result.ok and map_result.data is not None:
                        data = map_result.data
                        print_map_diagnostics(data)
                        diagnostic_frame = draw_map_preview(diagnostic_frame, data)
                        obstacles = sum(cell == 1 for row in data.grid for cell in row)
                        missing = " · 출발점 미검출" if data.start is None else ""
                        if data.goal is None:
                            missing += " · 도착점 미검출"
                        message = (f"맵 {data.rows}x{data.cols} · 장애물 {obstacles}개"
                                   f"{missing} · R: 실시간 화면")
                    else:
                        message = format_map_result(map_result)
                        print(message)
                except Exception as exc:
                    traceback.print_exc()
                    message = f"맵 인식 실행 오류: {type(exc).__name__}"
            elif key == ord("r"):
                diagnostic_frame = None
                message = None

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error as exc:
        print(f"OpenCV 화면 표시 오류: {exc}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
