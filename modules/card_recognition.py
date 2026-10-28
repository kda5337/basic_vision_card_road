
"""
============================================================
CardRoad - 카드 인식 모듈
담당: 송중현
============================================================

[역할]
웹캠 영상에서 명령어 카드를 인식하고 다음 정보를 반환합니다.

- 인식된 카드의 명령어 종류
- 카드의 배치 순서 (왼쪽 → 오른쪽)
- 카드 위치 및 인식 신뢰도 (선택 사항)

[인터페이스]
입력: OpenCV 이미지 (np.ndarray)
출력: DetectionResult[list[CardData]]

[공통 데이터 구조]
common/models.py의 다음 항목을 참고합니다.

[1] Command
    인식 가능한 6종류의 명령어

[2] DetectionResult
    인식 성공/실패 결과

[4] CardData
    각 카드의 명령어 및 부가 정보

[주의 사항]
1. 반환하는 카드 리스트는 반드시 왼쪽부터 오른쪽
   순서로 정렬되어 있어야 합니다.

2. 명령어 실행이나 반복문 해석은 수행하지 않습니다.
   카드 종류와 순서만 인식하면 됩니다.

3. bbox와 confidence는 선택 사항입니다.
   해당 정보가 없다면 None으로 설정할 수 있습니다.

4. 인식된 카드 후보의 기호를 판별할 수 없다면
   오류를 반환해야 합니다.

5. 카드 검출 및 인식 알고리즘은 자유롭게 구현합니다.
============================================================
"""

import numpy as np

from common.models import (
    Command,
    CardData,
    DetectionResult
)


def detect_cards(
    frame: np.ndarray
) -> DetectionResult[list[CardData]]:
    """
    입력된 웹캠 이미지에서 명령어 카드를 인식합니다.

    Args:
        frame (np.ndarray):
            OpenCV BGR 이미지

    Returns:
        DetectionResult[list[CardData]]:

            성공한 경우:
                data에 CardData 리스트를 저장합니다.
                리스트는 왼쪽에서 오른쪽 순서입니다.
                error_code는 None입니다.

            실패한 경우:
                data는 None입니다.
                error_code에 오류 원인을 저장합니다.

    [반환 예시]

    cards = [
        CardData(command=Command.FORWARD),
        CardData(command=Command.TURN_RIGHT),
        CardData(command=Command.FORWARD)
    ]

    return DetectionResult(
        data=cards,
        error_code=None
    )

    [오류 예시]
    - NO_CARDS
    - UNKNOWN_CARD
    - INVALID_CARD_LAYOUT

    위 오류 코드는 예시입니다.
    구체적인 오류 구분은 담당자가 결정할 수 있습니다.
    """

    # TODO: 카드 인식 알고리즘 구현
    raise NotImplementedError
