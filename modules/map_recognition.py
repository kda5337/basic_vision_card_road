
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

import numpy as np

from common.models import MapData, DetectionResult


def detect_map(
    frame: np.ndarray
) -> DetectionResult[MapData]:
    """
    입력된 웹캠 이미지에서 맵을 인식합니다.

    Args:
        frame:
            OpenCV BGR 이미지

    Returns:
        DetectionResult[MapData]:
            성공 시 MapData 반환
            실패 시 error_code 반환
    """

    # TODO: 맵 인식 알고리즘 구현
    raise NotImplementedError
