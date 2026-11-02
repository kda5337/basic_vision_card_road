"""공유된 인쇄물/촬영 규격에 맞춘 AR 검사. 생성 데이터이며 인식 정확도 검사가 아닙니다."""
import unittest
from unittest.mock import patch

import numpy as np

from common.models import DetectionResult, Direction
import main as integration
from examples.demo_ar import sample, synthetic_frame
from modules.ar_visualization import ARVisualizer


class SpecTests(unittest.TestCase):
    def test_each_render_uses_latest_frame_without_restarting_animation(self):
        """카메라 대신 서로 다른 배열을 전달해 영상 고정/재생 초기화 여부를 검사."""
        board, cards, result = sample()
        ar = ARVisualizer()
        first = np.full((600, 960, 3), (20, 30, 40), np.uint8)
        second = np.full((600, 960, 3), (70, 80, 90), np.uint8)
        with patch("modules.ar_visualization.monotonic", return_value=100.) as clock:
            ar.begin(board, cards, result)
            initial = ar.render(first)
            clock.return_value = 100.3
            latest = ar.render(second)
            # UI와 맵 바깥의 영상은 각 호출에서 받은 원본 프레임을 따라야 합니다.
            np.testing.assert_array_equal(initial[85, 30], [20, 30, 40])
            np.testing.assert_array_equal(latest[85, 30], [70, 80, 90])
            x, y = ar._project([[.5, 4.]])[0]  # 4행→3행 전진 중간 중심
            np.testing.assert_array_equal(latest[y, x], [255, 110, 30])
            self.assertFalse(ar.is_finished())
            self.assertEqual(ar._started_at, 100.)
        self.assertTrue(np.all(first == (20, 30, 40)))
        self.assertTrue(np.all(second == (70, 80, 90)))

    def test_start_again_uses_new_recognition_and_execution_results(self):
        """기존 process_game과 실제 AR을 연결하되 인식/엔진만 테스트 입력으로 대체."""
        old_map, old_cards, old_result = sample(scenario="collision")
        new_map, new_cards, new_result = sample(5, 7)
        frame_a = np.zeros((600, 960, 3), np.uint8)
        frame_b = np.full_like(frame_a, 30)
        ar = ARVisualizer()
        with patch.object(integration, "detect_map", side_effect=[DetectionResult(old_map), DetectionResult(new_map)]) as maps, \
             patch.object(integration, "detect_cards", side_effect=[DetectionResult(old_cards), DetectionResult(new_cards)]) as cards, \
             patch.object(integration, "execute", side_effect=[old_result, new_result]) as execute, \
             patch("modules.ar_visualization.monotonic", return_value=100.) as clock:
            self.assertIsNone(integration.process_game(frame_a, ar))
            clock.return_value = 102.
            self.assertTrue(ar.is_finished())
            self.assertIsNotNone(ar._result.attempted_cell)
            self.assertIsNone(integration.process_game(frame_b, ar))
            self.assertFalse(ar.is_finished())
            self.assertIsNone(ar._result.attempted_cell)
            self.assertIs(maps.call_args.args[0], frame_b)
            self.assertIs(cards.call_args.args[0], frame_b)
            self.assertIs(execute.call_args.args[0], new_map)
            self.assertEqual(execute.call_args.args[1], [card.command for card in new_cards])
            clock.return_value = 104.
            ar.render(frame_b)
            self.assertEqual(ar._pose(2.)[:2], new_map.goal)

    def test_documented_sizes_keep_square_cells_and_initial_up_pose(self):
        for rows, cols in [(5, 5), (5, 7), (7, 5), (9, 9)]:
            with self.subTest(rows=rows, cols=cols):
                board, cards, result = sample(rows, cols)
                horizontal = (board.corners[1, 0] - board.corners[0, 0]) / cols
                vertical = (board.corners[3, 1] - board.corners[0, 1]) / rows
                self.assertAlmostEqual(float(horizontal), float(vertical), places=4)
                self.assertEqual(result.steps[0].before.direction, Direction.UP)
                ar = ARVisualizer()
                frame = np.zeros((600, 960, 3), np.uint8)
                with patch("modules.ar_visualization.monotonic", return_value=100.):
                    ar.begin(board, cards, result)
                    rendered = ar.render(frame)
                center_x = board.corners[0, 0] + horizontal * .5
                center_y = board.corners[0, 1] + vertical * (rows - .5)
                # 작은 칸의 흰색 외곽선/앤티앨리어싱을 피해 위쪽 내부 색상을 확인합니다.
                np.testing.assert_array_equal(rendered[round(center_y - .12 * vertical), round(center_x)], [255, 110, 30])
                np.testing.assert_array_equal(rendered[round(center_y), round(center_x + .36 * horizontal)], [0, 0, 0])
                self.assertEqual(rendered.shape, frame.shape)
                self.assertFalse(np.any(frame))

    def test_touching_portrait_cards_below_map_keep_repeat_highlight(self):
        board, cards, result = sample(scenario="repeat")
        for index, card in enumerate(cards):
            x, y, width, height = card.bbox
            self.assertAlmostEqual(width / height, 35 / 45)
            self.assertGreater(y, board.corners[:, 1].max())
            if index:
                previous = cards[index - 1].bbox
                self.assertEqual(x, previous[0] + previous[2])
        ar = ARVisualizer()
        with patch("modules.ar_visualization.monotonic", return_value=100.) as clock:
            ar.begin(board, cards, result)
            for elapsed, index in [(.1, 1), (.5, 2), (.9, 1), (1.3, 2)]:
                clock.return_value = 100. + elapsed
                output = ar.render(synthetic_frame(board, cards))
                x, y, _, _ = cards[index].bbox
                np.testing.assert_array_equal(output[y, x], [100, 220, 255])


if __name__ == "__main__":
    unittest.main()
