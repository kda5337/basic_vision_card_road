
"""
============================================================
CardRoad - 명령어 처리 및 게임 실행 모듈
담당: 유병훈
============================================================

[역할]
인식된 명령어를 해석하여 캐릭터를 이동시키고,
각 명령어의 실행 기록과 최종 게임 결과를 반환합니다.

- 전진 및 좌우 회전 처리
- 반복문 처리
- 장애물 충돌 및 맵 이탈 판정
- 목적지 도달 여부 판정
- 명령어별 실행 기록 생성

[인터페이스]
입력:
    MapData
    list[Command]

출력:
    ExecutionResult

[공통 데이터 구조]
common/models.py의 다음 항목을 참고합니다.

[1] Direction, Command, GameStatus
    캐릭터 방향, 명령어 종류, 실행 결과 상태

[3] MapData
    맵의 크기, 장애물, 시작점 및 도착점

[5] CharacterState
    캐릭터의 위치 및 방향

[6] ExecutionStep
    각 명령어의 실행 전후 상태 기록

[7] ExecutionResult
    전체 게임 실행 결과

[게임 규칙]
1. 캐릭터는 start 위치에서 UP 방향으로 시작합니다.

2. FORWARD:
   현재 방향으로 한 칸 전진합니다.

3. TURN_LEFT / TURN_RIGHT:
   현재 위치에서 90도 회전합니다.

4. REPEAT_2 / REPEAT_3:
   LOOP_END까지의 내부 명령어를 지정 횟수만큼 반복합니다.

5. 반복문 중첩은 지원하지 않습니다.

6. 장애물 충돌 또는 맵 이탈 시 즉시 실패합니다.

7. 목적지에 도달하면 즉시 성공합니다.

8. 명령어 실행이 끝났는데 목적지에 도달하지 못하면
   실패로 처리합니다.

9. 잘못된 반복문 구조는 실행 전에 검증하며,
   문법 오류가 있으면 실행하지 않습니다.

10. 빈 반복 구간은 허용하지 않습니다.

[오류 코드]
ERROR (실행 기록 없음):
    START_NOT_FOUND: 시작점 미검출 (시작점과 도착점이 모두 없으면 우선 반환)
    GOAL_NOT_FOUND: 도착점 미검출
    INVALID_COMMAND: Command Enum이 아닌 카드 (해당 카드 인덱스)
    UNEXPECTED_LOOP_END: 시작 없는 반복 종료 (종료 카드 인덱스)
    NESTED_LOOP: 중첩 반복 (안쪽 시작 카드 인덱스)
    EMPTY_LOOP: 빈 반복 구간 (시작 카드 인덱스)
    UNCLOSED_LOOP: 닫히지 않은 반복 (시작 카드 인덱스)
    INVALID_MAP: 잘못된 격자 크기·값 또는 시작점·도착점
FAILURE:
    OBSTACLE_COLLISION: 장애물 충돌 (전진 카드 인덱스)
    OUT_OF_BOUNDS: 맵 이탈 (전진 카드 인덱스)
    GOAL_NOT_REACHED: 명령 종료 후 목적지 미도달
표식 미검출, INVALID_MAP, GOAL_NOT_REACHED의 error_card_index는 None입니다.
표식 미검출은 명령어 문법 검사 전에 처리합니다.
시작점이 없으면 final_state는 CharacterState(-1, -1, Direction.UP)입니다.
이는 미배치 표시값이며, 실제 맵 위치로 렌더링하지 않습니다.
도착점만 없으면 final_state는 인식된 시작점의 UP 상태를 유지합니다.

[주의 사항]
1. 이미지 인식 및 AR 애니메이션은 구현하지 않습니다.

2. 각 이동·회전 명령의 실행 기록을 저장해야 합니다.

3. ExecutionStep.card_index는 원본 명령어 배열의
   인덱스를 사용합니다.

4. REPEAT_2, REPEAT_3, LOOP_END 자체는
   ExecutionStep에 기록하지 않습니다.

5. 충돌하거나 맵 밖으로 이동하려는 명령은
   정상 실행 단계에 추가하지 않습니다.

6. 구체적인 명령어 해석 알고리즘은 자유롭게 구현합니다.
============================================================
"""

from common.models import (
    Direction,
    Command,
    GameStatus,
    MapData,
    CharacterState,
    ExecutionStep,
    ExecutionResult
)


_REPEAT_COUNTS = {Command.REPEAT_2: 2, Command.REPEAT_3: 3}
_FORWARD_OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _compile_commands(
    commands: list[Command]
) -> tuple[list[tuple[int, Command]], tuple[str, int] | None]:
    """전체 문법을 검증하고 원본 카드 인덱스를 유지한 실행 목록을 만듭니다."""
    program: list[tuple[int, Command]] = []
    loop_start: int | None = None

    for index, command in enumerate(commands):
        if not isinstance(command, Command):
            return [], ("INVALID_COMMAND", index)

        if command in _REPEAT_COUNTS:
            if loop_start is not None:
                return [], ("NESTED_LOOP", index)
            loop_start = index
        elif command == Command.LOOP_END:
            if loop_start is None:
                return [], ("UNEXPECTED_LOOP_END", index)
            if index == loop_start + 1:
                return [], ("EMPTY_LOOP", loop_start)

            body = [
                (body_index, commands[body_index])
                for body_index in range(loop_start + 1, index)
            ]
            program.extend(body * _REPEAT_COUNTS[commands[loop_start]])
            loop_start = None
        elif loop_start is None:
            program.append((index, command))

    if loop_start is not None:
        return [], ("UNCLOSED_LOOP", loop_start)

    return program, None


def _is_valid_map(map_data: MapData) -> bool:
    """게임에 필요한 논리 격자만 검사합니다. 영상 꼭짓점은 사용하지 않습니다."""
    if map_data.start is None or map_data.goal is None:
        return False
    if map_data.rows <= 0 or map_data.cols <= 0:
        return False
    if len(map_data.grid) != map_data.rows:
        return False
    if any(len(row) != map_data.cols for row in map_data.grid):
        return False
    if any(cell not in (0, 1) for row in map_data.grid for cell in row):
        return False

    for row, col in (map_data.start, map_data.goal):
        if not (0 <= row < map_data.rows and 0 <= col < map_data.cols):
            return False
        if map_data.grid[row][col] != 0:
            return False

    return True


def execute(
    map_data: MapData,
    commands: list[Command]
) -> ExecutionResult:
    """
    맵과 명령어 목록을 바탕으로 게임을 실행합니다.

    Args:
        map_data (MapData):
            맵 인식 모듈에서 생성된 맵 정보

        commands (list[Command]):
            왼쪽부터 오른쪽으로 정렬된 명령어 목록

    Returns:
        ExecutionResult:
            게임의 최종 상태와 실행 기록을 반환합니다.

    [입력 예시]

    commands = [
        Command.FORWARD,
        Command.TURN_RIGHT,
        Command.FORWARD
    ]

    [실행 과정 예시]

    초기 상태:
        위치 (4, 0)
        방향 UP

    첫 번째 명령: FORWARD
        (4, 0, UP) → (3, 0, UP)

    두 번째 명령: TURN_RIGHT
        (3, 0, UP) → (3, 0, RIGHT)

    세 번째 명령: FORWARD
        (3, 0, RIGHT) → (3, 1, RIGHT)

    [반환할 정보]

    status:
        SUCCESS / FAILURE / ERROR

    steps:
        실제 실행된 명령어들의 ExecutionStep 리스트

    final_state:
        마지막 정상 캐릭터 상태

    error_code:
        실패 또는 오류 원인 (성공 시 None)

    error_card_index:
        오류를 일으킨 원본 카드 인덱스

    attempted_cell:
        충돌 또는 맵 이탈을 시도한 좌표

    [표식 미검출]
    시작점 또는 도착점이 None이면 이동 없이 ERROR를 반환합니다.
    시작점이 없을 때의 final_state는 (-1, -1, UP) 미배치 표시값입니다.
    도착점만 없으면 final_state는 시작 위치와 UP 방향을 유지합니다.
    """

    if map_data.start is None:
        return ExecutionResult(
            status=GameStatus.ERROR,
            steps=[],
            final_state=CharacterState(-1, -1, Direction.UP),
            error_code="START_NOT_FOUND"
        )

    state = CharacterState(
        row=map_data.start[0],
        col=map_data.start[1],
        direction=Direction.UP
    )
    if map_data.goal is None:
        return ExecutionResult(
            status=GameStatus.ERROR,
            steps=[],
            final_state=state,
            error_code="GOAL_NOT_FOUND"
        )

    program, syntax_error = _compile_commands(commands)

    # 도착점에 일찍 도달할 수 있어도 전체 명령의 문법 검증이 우선입니다.
    if syntax_error is not None:
        error_code, error_card_index = syntax_error
        return ExecutionResult(
            status=GameStatus.ERROR,
            steps=[],
            final_state=state,
            error_code=error_code,
            error_card_index=error_card_index
        )

    if not _is_valid_map(map_data):
        return ExecutionResult(
            status=GameStatus.ERROR,
            steps=[],
            final_state=state,
            error_code="INVALID_MAP"
        )

    steps: list[ExecutionStep] = []
    # 시작점과 도착점이 같은 맵은 이동 없이 성공합니다.
    if (state.row, state.col) == map_data.goal:
        return ExecutionResult(GameStatus.SUCCESS, steps, state)

    for card_index, command in program:
        before = state

        if command == Command.TURN_LEFT:
            state = CharacterState(
                state.row, state.col, Direction((state.direction - 1) % 4)
            )
        elif command == Command.TURN_RIGHT:
            state = CharacterState(
                state.row, state.col, Direction((state.direction + 1) % 4)
            )
        else:  # 검증·전개된 목록에는 전진과 회전 명령만 남습니다.
            delta_row, delta_col = _FORWARD_OFFSETS[state.direction]
            next_row = state.row + delta_row
            next_col = state.col + delta_col

            if not (0 <= next_row < map_data.rows and 0 <= next_col < map_data.cols):
                error_code = "OUT_OF_BOUNDS"
            elif map_data.grid[next_row][next_col] == 1:
                error_code = "OBSTACLE_COLLISION"
            else:
                error_code = None

            if error_code is not None:
                return ExecutionResult(
                    status=GameStatus.FAILURE,
                    steps=steps,
                    final_state=state,
                    error_code=error_code,
                    error_card_index=card_index,
                    attempted_cell=(next_row, next_col)
                )

            state = CharacterState(next_row, next_col, state.direction)

        steps.append(ExecutionStep(card_index, command, before, state))

        if (state.row, state.col) == map_data.goal:
            return ExecutionResult(GameStatus.SUCCESS, steps, state)

    return ExecutionResult(
        status=GameStatus.FAILURE,
        steps=steps,
        final_state=state,
        error_code="GOAL_NOT_REACHED"
    )
