import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from examples.demo_ar import fit_photo, main, select_corners


class PhotoARTests(unittest.TestCase):
    def test_resize_preserves_corner_coordinates(self):
        frame = np.zeros((3000, 4000, 3), np.uint8)
        corners = np.float32([[100, 200], [3900, 200], [3900, 2800], [100, 2800]])
        resized, mapped = fit_photo(frame, corners, 1000, 600)
        self.assertEqual(resized.shape, (600, 800, 3))
        np.testing.assert_allclose(mapped, corners * .2)
        self.assertEqual(frame.shape, (3000, 4000, 3))

    def test_small_photo_not_enlarged(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        resized, corners = fit_photo(frame, None, 1200, 750)
        self.assertIs(resized, frame)
        self.assertIsNone(corners)

    def test_photo_cli_keeps_original_background(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "종이사진.png"
            output = Path(directory) / "결과.png"
            frame = np.full((600, 960, 3), (87, 123, 159), np.uint8)
            ok, encoded = cv2.imencode(".png", frame)
            self.assertTrue(ok)
            encoded.tofile(path)
            with patch("examples.demo_ar.synthetic_frame", side_effect=AssertionError("Fake background used")):
                main(["--image", str(path), "--rows", "5", "--cols", "6",
                      "--corners", "150", "100", "760", "145", "830", "510", "90", "480",
                      "--output", str(output)])
            rendered = cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(rendered.shape, frame.shape)
            np.testing.assert_array_equal(rendered[200:300, 400:500], frame[200:300, 400:500])
            # 실제 사진 위에 파란 삼각형이 합성됐는지 확인합니다.
            self.assertTrue(np.any(np.all(rendered == [255, 110, 30], axis=2)))

    def test_corner_selection_rejects_crossed_order_then_accepts_reset(self):
        callbacks = {}
        calls = iter([0, 1, 2, 3])

        def set_callback(window, callback):
            callbacks["click"] = callback

        def wait_key(delay):
            stage = next(calls)
            if stage == 0:
                for x, y in [(20, 60), (180, 180), (180, 60), (20, 180)]:
                    callbacks["click"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                return 13  # 잘못된 순서는 Enter로 확정되지 않아야 합니다.
            if stage == 1:
                return ord("r")
            if stage == 3:
                return 13
            for x, y in [(20, 60), (180, 60), (180, 180), (20, 180)]:
                callbacks["click"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return 0

        # 마우스 입력 이후 다음 프레임에서 유효성을 평가합니다.
        with patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.setMouseCallback", side_effect=set_callback), \
                patch("cv2.getWindowProperty", return_value=1), \
                patch("cv2.waitKey", side_effect=wait_key):
            corners = select_corners(np.zeros((200, 200, 3), np.uint8), "test")
        np.testing.assert_array_equal(corners, [[20, 60], [180, 60], [180, 180], [20, 180]])


if __name__ == "__main__":
    unittest.main()
