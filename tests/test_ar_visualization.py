"""시간을 고정하여 프레임 속도/실제 대기 시간에 의존하지 않는 테스트."""
import unittest
from unittest.mock import patch

import numpy as np

from common.models import (CharacterState, Command, Direction, ExecutionResult,
                           ExecutionStep, GameStatus)
from examples.demo_ar import LAYOUTS, SCENARIOS, sample, synthetic_frame
from modules.ar_visualization import ARVisualizer


class ARTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch("modules.ar_visualization.monotonic", return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.ar = ARVisualizer()
        self.board, self.cards, self.result = sample()

    def test_idle_and_frame_ownership(self):
        frame = synthetic_frame(self.board)
        original = frame.copy()
        self.assertTrue(self.ar.is_finished())
        self.ar.begin(self.board, self.cards, self.result)
        rendered = self.ar.render(frame)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(rendered.dtype, frame.dtype)
        self.assertFalse(np.shares_memory(rendered, frame))
        self.assertFalse(self.ar.is_finished())

    def test_perspective_and_varying_map_sizes(self):
        for rows, cols in [(2, 2), (3, 9), (10, 4)]:
            board, cards, result = sample(rows, cols)
            self.ar.begin(board, cards, result)
            points = self.ar._project([[0, 0], [cols, 0], [cols, rows], [0, rows]])
            np.testing.assert_allclose(points, board.corners, atol=1)
            self.assertEqual(self.ar.render(synthetic_frame(board)).shape, (600, 960, 3))

    def test_move_midpoint_and_frame_drop_completion(self):
        self.ar.begin(self.board, self.cards, self.result)
        row, col, angle, card = self.ar._pose(.3)
        self.assertAlmostEqual(row, 3.5)
        self.assertEqual(col, 0)
        self.assertEqual(card, 0)
        self.now.return_value = 110.
        self.assertTrue(self.ar.is_finished())
        self.assertEqual(self.ar._pose(10.)[:2], (3, 1))
        self.ar.render(synthetic_frame(self.board))

    def test_turn_across_zero_in_both_directions(self):
        for before, after, command, expected in [
                (Direction.UP, Direction.LEFT, Command.TURN_LEFT, -np.pi / 4),
                (Direction.LEFT, Direction.UP, Command.TURN_RIGHT, 7 * np.pi / 4)]:
            start = CharacterState(1, 1, before)
            end = CharacterState(1, 1, after)
            result = ExecutionResult(GameStatus.SUCCESS,
                                     [ExecutionStep(0, command, start, end)], end)
            self.ar.begin(self.board, [], result)
            row, col, angle, _ = self.ar._pose(.2)
            self.assertEqual((row, col), (1, 1))
            self.assertAlmostEqual(angle, expected)

    def test_failure_and_empty_error(self):
        for status in [GameStatus.FAILURE, GameStatus.ERROR]:
            final = CharacterState(0, 1, Direction.UP)
            result = ExecutionResult(status, [], final, "TEST_ERROR", 0, (-1, 1))
            self.ar.begin(self.board, self.cards, result)
            self.assertTrue(self.ar.is_finished())
            self.assertEqual(self.ar._pose(10.)[:2], (0, 1))
            self.ar.render(synthetic_frame(self.board))

    def test_button_bounds_running_and_restart(self):
        shape = (600, 960, 3)
        left, top, right, bottom = self.ar._button_rect(shape)
        self.assertTrue(self.ar.is_start_clicked(left, top, shape))
        self.assertFalse(self.ar.is_start_clicked(right, bottom, shape))
        self.ar.begin(self.board, self.cards, self.result)
        self.assertFalse(self.ar.is_start_clicked(left, top, shape))
        self.now.return_value = 110.
        self.assertTrue(self.ar.is_start_clicked(left, top, shape))
        self.ar.begin(self.board, self.cards, self.result)
        self.assertFalse(self.ar.is_finished())

    def test_snapshot_is_independent_of_recognition_updates(self):
        self.ar.begin(self.board, self.cards, self.result)
        self.board.corners[:] = 0
        self.result.steps.clear()
        self.cards.clear()
        self.assertFalse(self.ar.is_finished())
        self.assertEqual(len(self.ar._result.steps), 3)
        self.ar.render(np.zeros((600, 960, 3), dtype=np.uint8))

    def test_invalid_corners_and_frames(self):
        self.board.corners[:] = 0
        with self.assertRaises(ValueError):
            self.ar.begin(self.board, self.cards, self.result)
        for frame in [np.zeros((10, 10)), np.zeros((0, 10, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                self.ar.render(frame)

    def test_small_frame_and_korean_message(self):
        self.ar.render(np.zeros((20, 30, 3), dtype=np.uint8))
        frame = np.zeros((100, 400, 3), dtype=np.uint8)
        with patch.object(self.ar, "_label", wraps=self.ar._label) as label:
            rendered = self.ar.render(frame, "맵 인식 실패: 격자를 다시 확인해 주세요")
        self.assertTrue(any(call.args[1] == "맵 인식 실패: 격자를 다시 확인해 주세요"
                            for call in label.call_args_list))
        self.assertTrue(np.any(rendered[36:59, 12:390]))
        self.assertFalse(np.any(frame))

    def test_all_demo_scenarios_and_layouts(self):
        for scenario in SCENARIOS:
            for corners in LAYOUTS.values():
                with self.subTest(scenario=scenario, corners=corners):
                    board, cards, result = sample(3, 7, corners, scenario)
                    frame = synthetic_frame(board, cards)
                    self.now.return_value = 100.
                    self.ar.begin(board, cards, result)
                    self.ar.render(frame)
                    self.now.return_value = 120.
                    self.assertIs(self.ar.is_finished(), True)
                    final = self.ar.render(frame)
                    self.assertEqual(final.shape, frame.shape)
                    self.assertEqual(self.ar._pose(20.)[:2],
                                     (result.final_state.row, result.final_state.col))
                    self.assertEqual(board.grid[board.goal[0]][board.goal[1]], 0)

    def test_card_highlight_follows_original_index_in_repeat(self):
        board, cards, result = sample(scenario="repeat")
        frame = synthetic_frame(board, cards)
        self.ar.begin(board, cards, result)
        for time, index in [(.1, 1), (.5, 2), (.9, 1), (1.3, 2)]:
            self.now.return_value = 100. + time
            output = self.ar.render(frame)
            x, y, _, _ = cards[index].bbox
            np.testing.assert_array_equal(output[y, x], [100, 220, 255])

    def test_failure_marks_error_card_after_last_valid_step(self):
        board, cards, result = sample(scenario="collision")
        frame = synthetic_frame(board, cards)
        self.ar.begin(board, cards, result)
        self.now.return_value = 101.
        output = self.ar.render(frame)
        x, y, _, _ = cards[1].bbox
        np.testing.assert_array_equal(output[y, x], [100, 110, 255])
        self.assertEqual(self.ar._pose(1.)[:2], board.start)

    def test_triangle_is_drawn_at_cell_center(self):
        # 직사각형 2x2 맵의 시작 셀 중심은 (250, 400)입니다.
        board, cards, result = sample(2, 2, [[100, 100], [700, 100], [700, 500], [100, 500]])
        self.ar.begin(board, cards, result)
        output = self.ar.render(np.zeros((600, 960, 3), dtype=np.uint8))
        np.testing.assert_array_equal(output[400, 250], [255, 110, 30])
        # 전진 중간에는 셀 중심이 100픽셀 위로 이동합니다.
        self.now.return_value = 100.3
        output = self.ar.render(np.zeros((600, 960, 3), dtype=np.uint8))
        np.testing.assert_array_equal(output[300, 250], [255, 110, 30])
        np.testing.assert_array_equal(output[400, 250], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
