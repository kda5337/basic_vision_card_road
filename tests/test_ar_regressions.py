"""인식 실패/재실행/카메라 영역과 UI 겹침에 대한 회귀 검사."""
import unittest
from unittest.mock import patch

import numpy as np

import main as integration
from common.models import (CardData, CharacterState, Command, DetectionResult,
                           Direction, ExecutionResult, GameStatus, MapData)
from examples.demo_ar import sample
from modules.ar_visualization import ARVisualizer


class RegressionTests(unittest.TestCase):
    def test_result_banner_expires_five_seconds_after_first_display_and_restarts(self):
        frame = np.zeros((600, 960, 3), np.uint8)
        for scenario, title in [('success', '성공!'), ('collision', '실패')]:
            with self.subTest(scenario=scenario), patch('modules.ar_visualization.monotonic', return_value=100.) as clock:
                board, cards, result = sample(scenario=scenario)
                ar = ARVisualizer()
                ar.begin(board, cards, result)

                def visible_labels():
                    with patch.object(ar, '_label', wraps=ar._label) as labels:
                        ar.render(frame)
                    return [call.args[1] for call in labels.call_args_list]

                self.assertNotIn(title, visible_labels())
                # 완료 프레임을 늦게 받아도 실제 표시 후 5초 동안 보입니다.
                clock.return_value = 120.
                self.assertIn(title, visible_labels())
                clock.return_value = 124.999
                self.assertIn(title, visible_labels())
                clock.return_value = 125.
                labels = visible_labels()
                self.assertFalse(any('성공' in text or '실패' in text for text in labels))
                self.assertIn('시작', labels)
                self.assertIsNotNone(ar._result)
                clock.return_value = 130.
                self.assertNotIn(title, visible_labels())
                ar.begin(board, cards, result)
                self.assertNotIn(title, visible_labels())
                clock.return_value = 140.
                self.assertIn(title, visible_labels())

    def test_unplaced_or_invalid_character_is_not_drawn(self):
        board = MapData(5, 5, [[0]*5 for _ in range(5)], None, (0, 4),
                        np.float32([[200, 200], [600, 200], [600, 600], [200, 600]]))
        ar = ARVisualizer()
        for start, row, col in [(None, -1, -1), ((4, 0), -1, 0), ((4, 0), 5, 0)]:
            with self.subTest(start=start, row=row, col=col):
                board.start = start
                result = ExecutionResult(GameStatus.ERROR, [], CharacterState(row, col, Direction.UP), 'START_NOT_FOUND')
                ar.begin(board, [], result)
                output = ar.render(np.zeros((800, 800, 3), np.uint8))
                self.assertFalse(np.any(np.all(output == [255, 110, 30], axis=2)))

    def test_map_or_card_rescan_failure_clears_old_result(self):
        board, cards, result = sample()
        frame = np.zeros((600, 960, 3), np.uint8)
        ar = ARVisualizer()
        for failure in ('map', 'cards'):
            with self.subTest(failure=failure), patch('modules.ar_visualization.monotonic', return_value=100.) as clock:
                ar.begin(board, cards, result)
                clock.return_value = 110.
                with patch.object(integration, 'detect_map', return_value=(DetectionResult(error_code='GRID_NOT_FOUND') if failure == 'map' else DetectionResult(board))), \
                     patch.object(integration, 'detect_cards', return_value=DetectionResult(error_code='CARDS_NOT_FOUND')):
                    message = integration.process_game(frame, ar)
                with patch.object(ar, '_label', wraps=ar._label) as labels:
                    output = ar.render(frame, message)
                self.assertFalse(any('성공' in call.args[1] for call in labels.call_args_list))
                self.assertFalse(np.any(np.all(output == [255, 110, 30], axis=2)))
                self.assertIsNone(ar._result)
                self.assertTrue(ar.is_finished())
                # 에러 메시지가 사라져도 이전 실행이 돌아오지 않고, 새 실행은 시작할 수 있습니다.
                ar.render(frame)
                self.assertIsNone(ar._result)
                ar.begin(board, cards, result)
                self.assertFalse(ar.is_finished())

    def test_ordinary_message_does_not_reset_playback(self):
        board, cards, result = sample()
        ar = ARVisualizer()
        with patch('modules.ar_visualization.monotonic', return_value=100.):
            ar.begin(board, cards, result)
            ar.render(np.zeros((600, 960, 3), np.uint8), '테스트용 이동 기록')
            self.assertFalse(ar.is_finished())
            self.assertIsNotNone(ar._result)

    def test_error_text_matches_engine_error_categories(self):
        board, _, _ = sample()
        ar = ARVisualizer()
        for code in ('START_NOT_FOUND', 'GOAL_NOT_FOUND', 'INVALID_MAP', 'INVALID_COMMAND',
                     'UNEXPECTED_LOOP_END', 'NESTED_LOOP', 'EMPTY_LOOP', 'UNCLOSED_LOOP'):
            with self.subTest(code=code):
                result = ExecutionResult(GameStatus.ERROR, [], CharacterState(4, 0, Direction.UP), code)
                ar.begin(board, [], result)
                with patch.object(ar, '_label', wraps=ar._label) as labels:
                    ar.render(np.zeros((600, 960, 3), np.uint8))
                text = labels.call_args_list[0].args[1]
                self.assertNotIn(code, text)
                self.assertTrue(text.startswith('맵 정보 오류' if code in ('START_NOT_FOUND', 'GOAL_NOT_FOUND', 'INVALID_MAP') else '명령어 오류'))

    def test_full_frame_map_character_not_hidden_by_button(self):
        board = MapData(5, 5, [[0]*5 for _ in range(5)], (4, 4), (0, 0),
                        np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]))
        result = ExecutionResult(GameStatus.FAILURE, [], CharacterState(4, 4, Direction.UP), 'GOAL_NOT_REACHED')
        ar = ARVisualizer()
        ar.begin(board, [], result)
        output = ar.render(np.zeros((480, 640, 3), np.uint8))
        np.testing.assert_array_equal(output[432, 576], [255, 110, 30])
        left, top, right, bottom = ar._button_rect(output.shape)
        self.assertFalse(left <= 576 < right and top <= 432 < bottom)
        self.assertTrue(ar.is_start_clicked(left + 5, top + 5, output.shape))
        self.assertFalse(ar.is_start_clicked(576, 432, output.shape))

    def test_ui_avoids_map_and_card_when_empty_space_exists(self):
        board, _, result = sample()
        cards = [CardData(Command.FORWARD, (790, 520, 160, 80))]
        ar = ARVisualizer()
        ar.begin(board, cards, result)
        ar.render(np.zeros((600, 960, 3), np.uint8))
        button = ar._button_rect((600, 960, 3))
        self.assertEqual(ar._overlap_score(button, ar._occupied_regions()), 0)
        self.assertNotEqual(button, (804, 536, 944, 584))


if __name__ == '__main__':
    unittest.main()
