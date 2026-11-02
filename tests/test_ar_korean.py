import os
import unittest
from unittest.mock import patch

import numpy as np

from common.models import CharacterState, Direction, ExecutionResult, GameStatus, MapData
from modules.ar_visualization import ARVisualizer


class KoreanTextTests(unittest.TestCase):
    def setUp(self):
        self.ar = ARVisualizer()

    def test_hangul_glyphs_are_not_replacement_boxes(self):
        frames = []
        for text in ["가", "나", "?", "□"]:
            frame = np.zeros((60, 80, 3), np.uint8)
            self.ar._label(frame, text, (10, 35), 65, (255, 255, 255))
            self.assertTrue(np.any(frame))
            frames.append(frame)
        for i in range(len(frames)):
            for j in range(i):
                self.assertFalse(np.array_equal(frames[i], frames[j]))

    def test_color_is_bgr_and_long_text_stays_in_its_region(self):
        frame = np.zeros((100, 300, 3), np.uint8)
        self.ar._label(frame, "맵 인식 실패: 격자를 다시 확인해 주세요" * 4,
                       (10, 35), 150, (0, 0, 255))
        self.assertFalse(np.any(frame[:, :, :2]))
        self.assertTrue(np.any(frame[:, :, 2]))
        self.assertFalse(np.any(frame[:, 160:]))
        self.assertFalse(np.any(frame[40:]))

    def test_missing_font_gives_actionable_error(self):
        with patch.dict(os.environ, {"CARDROAD_FONT_PATH": "missing-korean-font.ttf"}):
            with self.assertRaisesRegex(RuntimeError, "CARDROAD_FONT_PATH"):
                ARVisualizer()

    def test_explicit_font_path(self):
        with patch.dict(os.environ, {"CARDROAD_FONT_PATH": self.ar._font_path}):
            other = ARVisualizer()
        self.assertEqual(other._font_path, self.ar._font_path)

    def test_dynamic_messages_use_bounded_cache(self):
        frame = np.zeros((100, 300, 3), np.uint8)
        for i in range(140):
            self.ar.render(frame, f"인식 상태 {i}")
        self.assertLessEqual(len(self.ar._text_masks), 128)

    def test_unknown_error_code_is_preserved(self):
        board = MapData(2, 2, [[0, 0], [0, 0]], (1, 0), (0, 1),
                        np.float32([[30, 80], [280, 80], [280, 240], [30, 240]]))
        result = ExecutionResult(GameStatus.ERROR, [], CharacterState(1, 0, Direction.UP),
                                 "새로운 오류: TEAM_CUSTOM_ERROR")
        self.ar.begin(board, [], result)
        with patch.object(self.ar, "_label", wraps=self.ar._label) as label:
            self.ar.render(np.zeros((300, 400, 3), np.uint8))
        self.assertIn("새로운 오류: TEAM_CUSTOM_ERROR", label.call_args_list[0].args[1])


if __name__ == "__main__":
    unittest.main()
