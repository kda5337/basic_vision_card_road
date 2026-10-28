
"""
============================================================
CardRoad - 메인 실행 및 시스템 통합 모듈
담당: 강호현
============================================================

[역할]
각 기능 모듈을 연결하고 전체 게임 실행 흐름을 관리합니다.

[실행 흐름]
1. 웹캠 영상 수집
2. 사용자 Start 버튼 입력
3. 맵 인식
4. 카드 인식
5. 명령어 추출 및 게임 실행
6. AR 애니메이션 재생
7. 최종 결과 표시

[주의 사항]
- 현재는 모듈들이 미구현 상태이므로 전체 게임은 실행되지 않습니다.
============================================================
"""

import numpy as np

from common.models import MapData, CardData

from modules.map_recognition import detect_map
from modules.card_recognition import detect_cards
from modules.command_processor import execute
from modules.ar_visualization import ARVisualizer


def process_game(
    frame: np.ndarray,
    ar: ARVisualizer
) -> str | None:
    """
    Start 버튼 입력 시 호출할 게임 처리 함수입니다.

    입력:
        frame: Start 시점의 원본 웹캠 프레임
        ar: AR 시각화 객체

    반환:
        성공적으로 AR 재생을 시작했다면 None
        인식 실패 시 오류 메시지 문자열
    """

    # [1] 맵 인식
    map_result = detect_map(frame)

    if not map_result.ok or map_result.data is None:
        return f"맵 인식 실패: {map_result.error_code}"

    map_data: MapData = map_result.data

    # [2] 카드 인식
    card_result = detect_cards(frame)

    if not card_result.ok or card_result.data is None:
        return f"카드 인식 실패: {card_result.error_code}"

    cards: list[CardData] = card_result.data

    # [3] 카드에서 명령어만 추출
    commands = [card.command for card in cards]

    # [4] 게임 실행 결과 계산
    result = execute(map_data, commands)

    # [5] AR 애니메이션 재생
    ar.begin(map_data, cards, result)

    return None


def main():
    """
    프로그램 시작점입니다.

    향후 구현할 내용:
    - 웹캠 초기화 및 프레임 수집
    - Start 버튼 이벤트 처리
    - process_game() 호출
    - ar.render()를 이용한 화면 출력
    - 애니메이션 완료 및 오류 상태 관리
    - 프로그램 종료 시 카메라 자원 해제
    """

    print("CardRoad 시스템 통합 모듈 준비 완료")

    # TODO: 웹캠 및 전체 실행 루프 구현


if __name__ == "__main__":
    main()
