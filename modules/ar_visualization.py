
"""
============================================================
CardRoad - AR 시각화 및 UI 모듈
담당: 강주원
============================================================

[역할]
게임 실행 결과를 실제 웹캠 영상 위에 시각화합니다.

- 파란색 삼각형 캐릭터 렌더링
- 캐릭터 전진 애니메이션
- 캐릭터 좌우 90도 회전 애니메이션
- 실제 격자 맵 위에 AR 합성
- Start 버튼 및 게임 상태 UI 표시
- 게임 성공/실패 메시지 표시

[인터페이스]
ARVisualizer 클래스를 통해 다음 기능을 제공합니다.

1. begin()
   실행 결과를 전달받아 애니메이션을 준비합니다.

2. render()
   현재 프레임에 AR 및 UI를 합성하여 반환합니다.

3. is_finished()
   애니메이션 재생 완료 여부를 반환합니다.

4. is_start_clicked()
   Start 버튼 클릭 여부를 판단합니다.

[공통 데이터 구조]
common/models.py의 다음 항목을 참고합니다.

[1] Direction, Command, GameStatus
    캐릭터 방향 및 실행 결과 해석

[3] MapData
    격자 크기 및 영상 내 맵 위치

[4] CardData
    명령어 카드 위치 정보 (선택)

[5] CharacterState
    캐릭터 위치 및 방향 정보

[6] ExecutionStep
    이동 및 회전 애니메이션 생성

[7] ExecutionResult
    전체 실행 기록 및 최종 결과

[주의 사항]
1. 게임 실행 로직은 직접 구현하지 않습니다.
   ExecutionResult에 기록된 결과만 시각화합니다.

2. MapData.corners를 활용하여 격자 좌표를
   영상 내 픽셀 좌표로 변환합니다.

3. 캐릭터의 초기 위치는 MapData.start입니다.

4. 이동 및 회전 애니메이션은 ExecutionStep의
   before와 after 상태를 바탕으로 구현합니다.

5. 애니메이션의 속도, 보간 방식, 내부 상태 관리 등은
   담당자가 자유롭게 구현합니다.

6. render()는 원본 프레임과 동일한 크기 및 형식의
   OpenCV BGR 이미지를 반환해야 합니다.

7. 실제 맵 인식이나 카드 인식은 수행하지 않습니다.

[인쇄물 및 촬영 규격과의 연결]
- 동일한 크기의 정사각형 칸을 기준으로 하며 rows/cols는 인식 결과를 사용합니다.
  5×5, 7×5, 9×9 등 특정 크기로 제한하지 않습니다.
- 초기 상태 Direction.UP은 명령 처리 모듈에서 생성합니다. AR은 전달받은
  before/after/final_state를 그대로 재생하며 회전 결과를 UP으로 덮어쓰지 않습니다.
- A4 가로 용지와 35×45mm 카드의 물리 크기를 고정 픽셀 값으로 바꾸지 않습니다.
  corners는 종이 모서리가 아닌 격자 외곽이며 bbox와 frame도 같은 픽셀 좌표계입니다.
- 카메라와 종이를 수직 촬영 위치에 고정합니다. begin 이후 좌표는 고정되므로
  카메라/종이를 이동하거나 프레임 크기를 변경했다면 다시 인식한 뒤 실행해야 합니다.
============================================================
"""

from collections import OrderedDict
from copy import deepcopy
import os
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.models import (
    MapData,
    CardData,
    ExecutionResult,
    Command,
    GameStatus,
)


class ARVisualizer:
    """
    AR 시각화 및 UI를 담당하는 클래스입니다.

    게임 실행 기록을 저장하고, 웹캠 프레임마다
    애니메이션을 갱신하여 화면에 표시합니다.

    내부 변수 및 보조 메서드는 자유롭게 정의할 수 있지만,
    아래에 선언된 공개 메서드의 입출력은 유지해야 합니다.
    """

    def __init__(self):
        """
        AR 시각화 객체를 초기화합니다.

        이곳에서 애니메이션 상태, UI 상태 등
        필요한 내부 변수를 초기화합니다.

        초기 상태에서는 애니메이션이 재생 중이지 않습니다.
        """

        self._map = None
        self._cards = []
        self._result = None
        self._transform = None
        self._ends = np.array([], dtype=float)
        self._started_at = 0.0
        self._font_path = self._find_font()
        self._fonts = {}
        self._text_masks = OrderedDict()

    def begin(
        self,
        map_data: MapData,
        cards: list[CardData],
        result: ExecutionResult
    ) -> None:
        """
        새로운 게임 실행 결과의 AR 재생을 시작합니다.

        Args:
            map_data (MapData):
                인식된 맵 정보

            cards (list[CardData]):
                인식된 카드 목록
                카드 위치 강조 기능에 선택적으로 사용

            result (ExecutionResult):
                게임 엔진에서 생성한 전체 실행 기록

        [처리할 작업]
        - 맵 정보 저장
        - 캐릭터 초기 위치 설정
        - 실행 단계 목록 저장
        - 애니메이션 재생 상태 초기화

        주의:
            명령어를 다시 해석하거나
            게임의 성공/실패를 재판정하지 않습니다.

            실행 결과에 steps가 없다면
            이동 애니메이션 없이 완료 처리할 수 있습니다.
        """

        corners = np.asarray(map_data.corners, dtype=np.float32)
        if map_data.rows <= 0 or map_data.cols <= 0:
            raise ValueError("Map rows and cols must be positive")
        if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                or not cv2.isContourConvex(corners)
                or abs(cv2.contourArea(corners)) < 1.0):
            raise ValueError("Map corners must form a non-degenerate convex quadrilateral")
        # 격자 평면에서 이동/회전한 뒤 원본 영상에 투영해야 원근이 유지됩니다.
        source = np.float32([[0, 0], [map_data.cols, 0],
                             [map_data.cols, map_data.rows], [0, map_data.rows]])
        transform = cv2.getPerspectiveTransform(source, corners)
        # 실행 도중 인식 모듈이 원래 객체를 갱신해도 재생 내용은 유지합니다.
        self._map, self._cards, self._result = deepcopy((map_data, cards, result))
        self._transform = transform
        durations = [0.6 if step.command == Command.FORWARD else 0.4
                     for step in self._result.steps]
        self._ends = np.cumsum(durations)
        self._started_at = monotonic()

    def render(
        self,
        frame: np.ndarray,
        message: str | None = None
    ) -> np.ndarray:
        """
        웹캠 영상에 AR 및 UI를 합성합니다.

        Args:
            frame (np.ndarray):
                현재 웹캠 프레임 (BGR)

            message (str | None):
                화면에 표시할 상태 메시지
                메시지가 없으면 None

        Returns:
            np.ndarray:
                AR 및 UI가 합성된 OpenCV BGR 이미지

        [처리할 작업]
        - 현재 애니메이션 진행 상태 갱신
        - 삼각형 캐릭터 렌더링
        - 이동 및 회전 애니메이션 표시
        - Start 버튼 및 상태 메시지 표시

        render()는 메인 루프에서 반복적으로 호출됩니다.

        프레임마다 캐릭터의 위치와 방향을 갱신하고,
        합성된 이미지를 반환해야 합니다.
        """

        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or min(frame.shape[:2]) < 1):
            raise ValueError("frame must be a non-empty uint8 BGR image")
        output = frame.copy()
        elapsed = max(0.0, monotonic() - self._started_at)
        running = bool(self._ends.size and elapsed < self._ends[-1])
        status = "시작 버튼을 눌러 주세요"
        color = (100, 210, 130)
        if self._result is not None:
            row, col, angle, card_index = self._pose(elapsed)
            # 위쪽을 향하는 파란 삼각형. 좌표 단위는 격자 한 칸입니다.
            local = np.float32([[0, -0.34], [-0.25, 0.25], [0.25, 0.25]])
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            points = local @ rotation.T + [col + 0.5, row + 0.5]
            polygon = self._project(points)
            cv2.fillConvexPoly(output, polygon, (255, 110, 30), cv2.LINE_AA)
            cv2.polylines(output, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
            if running:
                step_index = int(np.searchsorted(self._ends, elapsed, side="right"))
                step = self._result.steps[step_index]
                command_text = {
                    Command.FORWARD: "전진",
                    Command.TURN_LEFT: "왼쪽 회전",
                    Command.TURN_RIGHT: "오른쪽 회전",
                }.get(step.command, step.command.value)
                status = f"{step_index + 1}/{len(self._result.steps)}단계 · {command_text}"
                color = (100, 220, 255)
            else:
                status = {
                    GameStatus.SUCCESS: "성공 · 도착점에 도착했습니다",
                    GameStatus.FAILURE: "실패",
                    GameStatus.ERROR: "명령어 오류",
                }[self._result.status]
                if self._result.error_code:
                    descriptions = {
                        "OBSTACLE_COLLISION": "장애물과 충돌했습니다",
                        "OUT_OF_BOUNDS": "맵 밖으로 이동할 수 없습니다",
                        "GOAL_NOT_REACHED": "도착점에 도달하지 못했습니다",
                        "UNCLOSED_LOOP": "반복문을 닫는 카드가 없습니다",
                    }
                    code = self._result.error_code
                    status += f": {descriptions.get(code, code)}"
                color = ((100, 220, 130) if self._result.status == GameStatus.SUCCESS
                         else (100, 110, 255))
                cell = self._result.attempted_cell
                if cell is not None:
                    r, c = cell
                    # 실패 위치만 표시하며 캐릭터는 final_state에 유지합니다.
                    cross = self._project([[c + .25, r + .25], [c + .75, r + .75],
                                           [c + .75, r + .25], [c + .25, r + .75]])
                    cv2.line(output, tuple(cross[0]), tuple(cross[1]), color, 3, cv2.LINE_AA)
                    cv2.line(output, tuple(cross[2]), tuple(cross[3]), color, 3, cv2.LINE_AA)
            if card_index is not None and 0 <= card_index < len(self._cards):
                bbox = self._cards[card_index].bbox
                if bbox is not None:
                    x, y, w, h = map(int, bbox)
                    if w > 0 and h > 0:
                        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), color, 3)
        self._draw_ui(output, status, message, running, color)
        return output

    def is_finished(self) -> bool:
        """
        현재 애니메이션의 재생 완료 여부를 반환합니다.

        Returns:
            bool:
                True  - 애니메이션이 종료되었거나 재생 중이 아님
                False - 애니메이션이 재생 중임

        통합 모듈은 이 값을 확인하여
        게임 종료 메시지 표시 시점을 결정합니다.
        """

        return bool(not self._ends.size or monotonic() - self._started_at >= self._ends[-1])

    def is_start_clicked(
        self,
        x: int,
        y: int,
        frame_shape: tuple[int, ...]
    ) -> bool:
        """
        사용자가 Start 버튼을 클릭했는지 판단합니다.

        Args:
            x (int):
                클릭한 위치의 영상 x좌표

            y (int):
                클릭한 위치의 영상 y좌표

            frame_shape (tuple[int, ...]):
                현재 화면의 이미지 크기
                일반적으로 (height, width, channels)

        Returns:
            bool:
                True  - Start 버튼 영역을 클릭한 경우
                False - 그 외의 경우

        [주의 사항]
        마우스 이벤트 자체는 통합 모듈에서 전달합니다.

        Start 버튼의 위치, 크기, 디자인 등은
        AR/UI 담당자가 자유롭게 결정합니다.

        클릭 좌표는 render()에 전달되는 영상과
        동일한 픽셀 좌표계를 사용해야 합니다.
        """

        left, top, right, bottom = self._button_rect(frame_shape)
        return self.is_finished() and left <= x < right and top <= y < bottom

    def _pose(self, elapsed):
        """격자 좌표에서 보간합니다. 방향 3→0도 90도만 회전합니다."""
        index = int(np.searchsorted(self._ends, elapsed, side="right"))
        if index >= len(self._result.steps):
            state = self._result.final_state
            return (state.row, state.col, float(state.direction) * np.pi / 2,
                    self._result.error_card_index)
        step = self._result.steps[index]
        start = self._ends[index - 1] if index else 0.0
        progress = np.clip((elapsed - start) / (self._ends[index] - start), 0, 1)
        progress = progress * progress * (3 - 2 * progress)  # 부드러운 가감속
        before, after = step.before, step.after
        turn = (int(after.direction) - int(before.direction) + 2) % 4 - 2
        row = before.row + (after.row - before.row) * progress
        col = before.col + (after.col - before.col) * progress
        angle = (int(before.direction) + turn * progress) * np.pi / 2
        return row, col, angle, step.card_index

    def _project(self, points):
        points = np.asarray(points, np.float32).reshape(1, -1, 2)
        pixels = cv2.perspectiveTransform(points, self._transform)[0]
        return np.rint(pixels).astype(np.int32)

    @staticmethod
    def _button_rect(frame_shape):
        height, width = frame_shape[:2]
        margin = max(0, min(16, width // 20, height // 20))
        button_width = min(140, max(1, width - 2 * margin))
        button_height = min(48, max(1, height - 2 * margin))
        right, bottom = width - margin, height - margin
        return right - button_width, bottom - button_height, right, bottom

    @staticmethod
    def _find_font():
        """운영체제의 한글 글꼴 사용. 글꼴 파일은 제출물에 복사하지 않습니다."""
        override = os.environ.get("CARDROAD_FONT_PATH")
        candidates = ([Path(override)] if override else [
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf",
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ])
        for path in candidates:
            if not path.is_file():
                continue
            try:
                font = ImageFont.truetype(str(path), 20)
                # 한글이 없는 글꼴은 두 글자가 같은 대체 사각형으로 그려집니다.
                first, last = font.getmask("가"), font.getmask("힣")
                if (first.size, bytes(first)) != (last.size, bytes(last)):
                    return str(path)
            except OSError:
                continue
        raise RuntimeError(
            "한글 글꼴을 찾지 못했습니다. 맑은 고딕/Noto Sans CJK/NanumGothic을 설치하거나 "
            "CARDROAD_FONT_PATH 환경 변수에 한글 TTF/OTF/TTC 파일 경로를 지정해 주세요."
        )

    def _text_mask(self, text, width, size):
        """텍스트 영역만 그려 캐시합니다. 매 프레임 전체 영상을 PIL로 바꾸지 않습니다."""
        text = " ".join(text.split())
        key = (text, width, size)
        if key in self._text_masks:
            self._text_masks.move_to_end(key)
            return self._text_masks[key]
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self._font_path, size)
        font = self._fonts[size]
        if font.getlength(text) > width:
            suffix = "…"
            if font.getlength(suffix) > width:
                return None
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if font.getlength(text[:middle] + suffix) <= width:
                    low = middle
                else:
                    high = middle - 1
            text = text[:low] + suffix
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        if right <= left or bottom <= top:
            return None
        mask = Image.new("L", (right - left, bottom - top), 0)
        ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255, anchor="ls")
        result = (left, top, np.asarray(mask))
        self._text_masks[key] = result
        if len(self._text_masks) > 128:
            self._text_masks.popitem(last=False)
        return result

    def _label(self, frame, text, position, width, color, scale=.6):
        if width <= 4 or not text:
            return
        mask_data = self._text_mask(text, width, max(12, round(scale * 34)))
        if mask_data is None:
            return
        left, top, mask = mask_data
        x, y = position[0] + left, position[1] + top
        # 화면 바깥으로 나가는 부분은 안전하게 잘라냅니다.
        x0, y0 = max(0, x), max(0, y)
        x1 = min(frame.shape[1], x + mask.shape[1], position[0] + width)
        y1 = min(frame.shape[0], y + mask.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        alpha = mask[y0-y:y1-y, x0-x:x1-x, None].astype(np.float32) / 255.
        region = frame[y0:y1, x0:x1]
        # color는 원래 OpenCV와 같은 BGR 순서입니다.
        region[:] = np.rint(region * (1 - alpha) + np.asarray(color) * alpha).astype(np.uint8)

    def _draw_ui(self, frame, status, message, running, color):
        height, width = frame.shape[:2]
        panel_height = min(height, 72 if message else 42)
        panel = frame[:panel_height]
        panel[:] = (panel.astype(np.float32) * .25).astype(np.uint8)
        self._label(frame, status, (12, 27), width - 24, color)
        if message:
            self._label(frame, message, (12, 56), width - 24, (240, 240, 240), .5)
        left, top, right, bottom = self._button_rect(frame.shape)
        cv2.rectangle(frame, (left, top), (right - 1, bottom - 1),
                      (90, 90, 90) if running else (70, 150, 40), -1)
        label = "재생 중" if running else "시작"
        self._label(frame, label, (left + 12, top + (bottom - top) // 2 + 7),
                    right - left - 24, (255, 255, 255))
