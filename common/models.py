
"""
============================================================
CardRoad - 공통 데이터 구조 및 인터페이스 정의
============================================================

이 파일은 모든 모듈이 공통으로 사용하는 데이터 구조를 정의합니다.

각 팀원은 자신에게 해당하는 번호의 내용을 확인하고,
정의된 클래스를 import하여 사용하면 됩니다.

[담당자별 참고 항목]

각 팀원은 아래 번호의 데이터 구조를 참고하세요.

1. 김단아 - 맵 인식
   - [2] DetectionResult : 인식 결과 반환 (생성)
   - [3] MapData         : 맵 구조 및 위치 정보 (생성)

2. 송중현 - 카드 인식
   - [1] Command         : 인식한 카드 종류 지정 (사용)
   - [2] DetectionResult : 인식 결과 반환 (생성)
   - [4] CardData        : 카드 정보 저장 (생성)

3. 유병훈 - 게임 실행
   - [1] Enum            : 명령어, 방향, 결과 상태 (사용)
   - [3] MapData         : 이동 가능 여부 판단 (입력)
   - [5] CharacterState  : 캐릭터 상태 관리 (생성)
   - [6] ExecutionStep   : 명령어 실행 과정 기록 (생성)
   - [7] ExecutionResult : 최종 게임 결과 반환 (생성)

4. 강주원 - AR 및 UI
   - [1] Enum            : 방향 및 명령어, 결과 해석 (필요시)
   - [3] MapData         : 격자와 픽셀 좌표 변환 (입력)
   - [4] CardData        : 카드 위치 강조 (선택)
   - [5] CharacterState  : 캐릭터 위치 및 방향 확인 (입력)
   - [6] ExecutionStep   : 이동·회전 애니메이션 (입력)
   - [7] ExecutionResult : 실행 결과 및 메시지 표시 (입력)

5. 강호현 - 시스템 통합
   - [1]~[7] 전체 데이터 구조 참고
   - 각 모듈의 결과를 다음 모듈에 전달
   - 인식 오류 및 실행 흐름 관리

[용어 정의]
- 생성: 자신의 모듈에서 데이터를 만들어 반환해야하는 것입니다.
- 입력: 다른 모듈에서 전달받아 사용하는 것입니다.
- 선택: 필요 시 사용할 수도 있는 데이터입니다.

[공통 규칙]
- 격자 좌표는 (row, col) 순서로 표현합니다.
- 좌표는 0부터 시작하며, 왼쪽 위 칸이 (0, 0)입니다.
- row는 아래쪽으로, col은 오른쪽으로 증가합니다.
- 영상 좌표는 OpenCV의 (x, y) 픽셀 좌표를 사용합니다.
- 함수와 클래스의 입출력 형식은 임의로 변경하지 않습니다.
- 각 모듈의 내부 구현 방식은 담당자가 자유롭게 결정합니다.
- 공통 데이터 구조 변경이 필요하면 통합 담당자와 협의합니다.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Generic, TypeVar

import numpy as np


# ============================================================
# 1. 공통 열거형 (Enums)
# ============================================================

class Direction(IntEnum):
    """
    캐릭터가 바라보는 방향을 나타냅니다.

    방향은 위쪽을 기준으로 시계 방향으로 정의합니다.

         UP(0)
           ↑
  (3)LEFT ← → RIGHT(1)
           ↓
         DOWN(2)

    IntEnum을 사용하므로 정수 연산이 가능합니다.

    예시:
        오른쪽 회전: Direction((direction + 1) % 4)
        왼쪽 회전:   Direction((direction - 1) % 4)

    캐릭터의 초기 방향은 UP입니다.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Command(str, Enum):
    """
    인식 가능한 명령어 카드의 종류를 정의합니다.

    총 6종류의 명령어를 지원합니다.

    [이동 명령어]
    FORWARD:
        현재 바라보는 방향으로 한 칸 전진합니다.

    TURN_LEFT:
        현재 위치에서 왼쪽으로 90도 회전합니다.

    TURN_RIGHT:
        현재 위치에서 오른쪽으로 90도 회전합니다.

    [반복 제어 명령어]
    REPEAT_2:
        LOOP_END까지의 내부 명령어를 2회 반복합니다.

    REPEAT_3:
        LOOP_END까지의 내부 명령어를 3회 반복합니다.

    LOOP_END:
        반복문의 끝을 나타냅니다.

    실제 반복문 해석은 Command Processing 모듈에서 수행합니다.

    현재 버전에서는 중첩 반복문을 지원하지 않습니다.
    """

    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"

    REPEAT_2 = "REPEAT_2"
    REPEAT_3 = "REPEAT_3"
    LOOP_END = "LOOP_END"


class GameStatus(str, Enum):
    """
    게임 실행의 최종 상태를 나타냅니다.

    SUCCESS:
        캐릭터가 목적지에 정상적으로 도착한 경우

    FAILURE:
        명령어 실행 자체는 가능했지만 게임에 실패한 경우
        예) 장애물 충돌, 맵 이탈, 목적지 미도달

    ERROR:
        명령어 구조가 잘못되어 정상적인 실행이 불가능한 경우
        예) 닫히지 않은 반복문, 잘못된 반복문 구조
    """

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"


# ============================================================
# 2. 인식 결과 공통 형식 (DetectionResult)
# ============================================================

# T는 데이터의 타입을 나타내는 제네릭 타입 변수입니다.
# MapData 또는 list[CardData]를 대신합니다.
T = TypeVar("T")


@dataclass
class DetectionResult(Generic[T]):
    """
    맵 인식과 카드 인식 모듈이 공통으로 사용하는 반환 형식입니다.

    [필드]
    data:
        인식에 성공했을 때 반환하는 데이터입니다.

        맵 인식의 경우: MapData
        카드 인식의 경우: list[CardData]

        인식에 실패했다면 None을 사용합니다.

    error_code:
        인식 실패 원인을 나타내는 문자열입니다.

        성공한 경우: None
        실패한 경우: 오류 코드 문자열

    [반환 예시]

    성공:
        DetectionResult(
            data=map_data,
            error_code=None
        )

    실패:
        DetectionResult(
            data=None,
            error_code="MAP_NOT_FOUND"
        )

    [주의 사항]
    - 성공 시 data에는 유효한 결과가 있어야 합니다.
    - 실패 시 data=None으로 반환합니다.
    - data와 error_code를 동시에 지정하지 않는 것을 원칙으로 합니다.
    """

    # 인식된 데이터. 기본값은 None입니다.
    data: T | None = None

    # 오류 코드. 오류가 없으면 None입니다.
    error_code: str | None = None

    @property
    def ok(self) -> bool:
        """
        인식 성공 여부를 반환합니다.

        data가 존재하고 error_code가 None이면 True입니다.

        사용 예시:
            result = detect_map(frame)

            if result.ok:
                map_data = result.data
            else:
                print(result.error_code)
        """

        return (
            self.data is not None
            and self.error_code is None
        )


# ============================================================
# 3. 맵 인식 결과 (MapData)
# 담당: 김단아
# ============================================================

@dataclass(eq=False)
class MapData:
    """
    Map Recognition 모듈이 반환하는 맵 정보입니다.

    웹캠 영상에서 맵을 인식한 뒤, 맵의 논리적 구조와
    실제 영상에서의 위치 정보를 저장합니다.

    [참고 사항]
    rows, cols, grid, start, goal은 게임 엔진에서 사용하며,
    corners는 주로 AR 시각화에서 사용합니다.

    eq=False는 NumPy 배열의 비교 문제를 피하기 위해
    dataclass의 자동 동등성 비교를 비활성화한 것입니다.
    """

    # 맵의 행, 열 개수
    rows: int
    cols: int

    # 장애물 배치를 나타내는 2차원 리스트
    # grid[row][col] 형태로 접근합니다.
    #
    # 0: 이동 가능한 칸
    # 1: 장애물
    #
    # Start와 Goal은 이동 가능한 칸이므로 0으로 표시합니다.
    # 시작점(Start)과 도착점(Goal)은 별도의 start, goal 필드에서
    # 좌표를 관리하므로 grid에 2 또는 3으로 표시할 필요가 없습니다.
    #
    # 리스트 크기는 반드시 rows x cols여야 합니다.
    #
    # 예시:
    # [
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ]
    grid: list[list[int]]

    # 캐릭터의 시작, 도착 위치 (row, col)
    start: tuple[int, int]
    goal: tuple[int, int]

    # 원본 웹캠 영상에서의 격자 외곽 네 꼭짓점
    #
    # 반드시 다음 순서로 저장합니다. (시계 방향)
    # 0: 왼쪽 위 (Top Left)
    # 1: 오른쪽 위 (Top Right)
    # 2: 오른쪽 아래 (Bottom Right)
    # 3: 왼쪽 아래 (Bottom Left)
    #
    # 배열 크기: (4, 2)
    # 권장 dtype: np.float32
    #
    # 예시:
    # np.array([
    #     [100, 100],
    #     [600, 100],
    #     [600, 600],
    #     [100, 600]
    # ], dtype=np.float32)
    #
    # 주의:
    # 종이 전체의 꼭짓점이 아니라
    # 실제 격자 외곽선의 꼭짓점을 의미합니다.
    #
    # AR 모듈은 이 좌표를 이용하여 격자 좌표를
    # 웹캠 이미지상의 픽셀 좌표로 변환합니다.
    corners: np.ndarray


# ============================================================
# 4. 카드 인식 결과 (CardData)
# 담당: 송중현
# ============================================================

@dataclass
class CardData:
    """
    Card Recognition 모듈이 인식한 카드 "하나"의 정보입니다.

    카드 인식 모듈은 화면에서 인식한 카드들을
    왼쪽부터 오른쪽 순서대로 정렬하여 반환해야 합니다.

    반환 형식:
        DetectionResult[list[CardData]]

    [필드]
    command:
        인식한 명령어의 종류

    bbox:
        카드가 위치한 영역의 바운딩 박스

    confidence:
        카드 인식 신뢰도

    [주의 사항]
    카드 인식 담당자는 명령어를 실행하거나
    반복문 구조를 해석할 필요가 없습니다.

    카드의 종류와 순서만 정확하게 반환하면 됩니다.

    bbox와 confidence를 계산하는 것은 필수가 아닙니다.
    해당 정보가 없으면 None을 반환해도 됩니다.

    단, 인식에 실패한 카드를 조용히 누락해서는 안 됩니다.
    확인된 카드 후보를 분류하지 못했다면
    인식 실패로 처리해야 합니다.
    """

    # 인식한 카드의 명령어 종류
    # 반드시 Command 열거형을 사용합니다.
    command: Command

    # ===카드의 원본 영상 내 위치===
    # 형식: (x, y, width, height)
    # x, y: 카드 바운딩 박스의 왼쪽 위 픽셀 좌표
    # width: 카드 영역의 가로 길이
    # height: 카드 영역의 세로 길이
    # 위치를 제공하지 않는 경우 None
    bbox: tuple[int, int, int, int] | None = None

    # ===카드 인식 신뢰도===
    # 값을 제공할 경우 0.0 ~ 1.0 범위를 사용합니다.
    # 값이 클수록 인식 신뢰도가 높음을 의미합니다.
    # 신뢰도를 제공하지 않는 경우 None
    confidence: float | None = None


# ============================================================
# 5. 캐릭터 상태 (CharacterState)
# 담당: 유병훈 / 강주원 공동 사용
# ============================================================

@dataclass(frozen=True)
class CharacterState:
    """
    캐릭터의 논리적인 위치와 방향을 나타냅니다.

    [필드]
    row:
        캐릭터가 위치한 행

    col:
        캐릭터가 위치한 열

    direction:
        캐릭터가 바라보는 방향

    [예시]
        CharacterState(
            row=4,
            col=0,
            direction=Direction.UP
        )

    [frozen=True 사용 이유]
    생성된 캐릭터 상태를 수정할 수 없도록 하여
    이전 실행 기록이 실수로 변경되는 것을 방지합니다.

    캐릭터가 움직였다면 기존 객체를 변경하지 말고
    새로운 CharacterState 객체를 생성합니다.

    [주의 사항]
    이 클래스는 실제 영상의 픽셀 위치가 아니라
    맵 내부의 격자 위치를 나타냅니다.

    AR 애니메이션의 중간 위치는 AR 모듈 내부에서
    별도로 계산합니다.
    """

    row: int
    col: int
    direction: Direction


# ============================================================
# 6. 개별 명령어 실행 기록 (ExecutionStep)
# 담당: 유병훈 생성 / 강주원 사용
# ============================================================

@dataclass
class ExecutionStep:
    """
    실제로 실행된 명령어 하나의 상태 변화를 기록합니다.

    게임 엔진은 명령어를 실행할 때마다
    ExecutionStep을 생성하여 실행 순서대로 저장합니다.

    AR 모듈은 이 기록을 읽어 캐릭터를 애니메이션으로
    이동시키거나 회전시킵니다.

    [필드]
    card_index:
        원본 명령어 배열에서 해당 명령어의 인덱스

    command:
        실제로 실행된 명령어

    before:
        명령어 실행 직전의 캐릭터 상태

    after:
        명령어 실행 직후의 캐릭터 상태

    [반복문 처리 규칙]
    REPEAT_2, REPEAT_3, LOOP_END는
    캐릭터를 직접 움직이지 않으므로 기록하지 않습니다.

    반복문 내부의 이동 또는 회전 명령어는
    실제 실행된 횟수만큼 기록합니다.

    예를 들어:
        [REPEAT_2, FORWARD, LOOP_END]

    실행 기록은:
        FORWARD (card_index=1)
        FORWARD (card_index=1)

    형태가 됩니다.

    [충돌 처리]
    충돌이나 맵 이탈을 일으킨 전진 명령은
    정상적인 ExecutionStep으로 기록하지 않습니다.

    오류 발생 정보는 ExecutionResult에서 관리합니다.
    """

    # 원본 명령어 배열의 인덱스 (0부터 시작)
    card_index: int

    # 실행한 명령어
    # FORWARD, TURN_LEFT, TURN_RIGHT 중 하나
    command: Command

    # 실행 직전 상태
    before: CharacterState

    # 실행 직후 상태
    after: CharacterState


# ============================================================
# 7. 전체 게임 실행 결과 (ExecutionResult)
# 담당: 유병훈 생성 / 강주원 및 강호현 사용
# ============================================================

@dataclass
class ExecutionResult:
    """
    전체 명령어 실행 결과를 저장합니다.

    Command Processing 모듈의 최종 반환 형식입니다.

    게임 엔진은 모든 명령어를 해석하고
    각 단계의 실행 기록과 최종 결과를 반환합니다.

    [필드]
    status:
        게임의 최종 상태

    steps:
        실제로 실행된 명령어들의 기록

    final_state:
        마지막 정상 캐릭터 상태

    error_code:
        실패 또는 오류의 구체적인 원인

    error_card_index:
        오류가 발생한 원본 명령어 인덱스

    attempted_cell:
        충돌하거나 이탈하려고 시도한 격자 좌표

    [실행 규칙]
    - 목적지에 도달하면 즉시 SUCCESS 처리합니다.
    - 장애물과 충돌하면 FAILURE 처리합니다.
    - 맵 밖으로 이동하면 FAILURE 처리합니다.
    - 명령어가 끝나도 목적지에 도달하지 못하면
      FAILURE 처리합니다.
    - 잘못된 반복문 구조는 ERROR 처리합니다.

    [실패 처리]
    장애물 충돌 또는 맵 이탈이 발생했다면,
    final_state는 충돌 직전의 정상 위치를 유지합니다.

    attempted_cell에는 이동하려던 좌표를 저장합니다.

    [ERROR 처리]
    명령어 문법 검증에서 오류가 발생했다면
    실제 캐릭터 이동을 시작하지 않습니다.

    이 경우:
        steps = []
        final_state = 초기 캐릭터 상태

    [AR 모듈과의 관계]
    AR 모듈은 steps를 순서대로 재생합니다.

    모든 애니메이션이 종료되면
    status와 error_code를 활용하여
    최종 게임 결과를 화면에 표시합니다.
    """

    # 게임의 최종 상태
    # SUCCESS / FAILURE / ERROR
    status: GameStatus

    # 실제 실행된 명령어 기록
    # 순서는 실행 순서와 동일해야 합니다.
    steps: list[ExecutionStep]

    # 마지막으로 정상적으로 도달한 상태
    final_state: CharacterState

    # 오류 원인
    # 예: "OBSTACLE_COLLISION", "UNCLOSED_LOOP"
    # 정상 성공한 경우 None
    error_code: str | None = None

    # 오류를 발생시킨 원본 카드의 인덱스
    # 특정 카드와 관련 없는 오류라면 None
    error_card_index: int | None = None

    # 충돌 또는 맵 이탈을 시도한 칸의 좌표 (row, col)
    # 해당하지 않는 경우 None
    attempted_cell: tuple[int, int] | None = None
