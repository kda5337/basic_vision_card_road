import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from examples.grid_size import GridSizeError, detect_grid_size
from examples.demo_ar import fit_photo, main, sample


def grid_photo(rows, cols, angled=True, shadow=False, missing_line=False):
    """크기 정답을 알고 있는 격자 촬영 시뮬레이션입니다. 실제 촬영 사진은 아닙니다."""
    image = np.full((801, 1001, 3), 245, np.uint8)
    for row in range(rows + 1):
        if missing_line and row == 1:
            continue
        y = round(row * 800 / rows)
        cv2.line(image, (0, y), (1000, y), (35, 35, 35), 4)
    for col in range(cols + 1):
        x = round(col * 1000 / cols)
        cv2.line(image, (x, 0), (x, 800), (35, 35, 35), 4)
    # 셀 안의 글자와 작은 검은 도형은 격자선으로 세면 안 됩니다.
    cv2.putText(image, "START", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (10, 10, 10), 2)
    cv2.rectangle(image, (20, 55), (int(1000 / cols) - 20, int(800 / rows) - 20), (20, 20, 20), -1)
    if shadow:
        light = np.linspace(.45, 1., 1001)[None, :, None]
        image = (image * light).astype(np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), .8)
    source = np.float32([[0, 0], [1000, 0], [1000, 800], [0, 800]])
    corners = (np.float32([[220, 100], [1010, 180], [1120, 850], [60, 790]])
               if angled else np.float32([[100, 100], [1100, 100], [1100, 900], [100, 900]]))
    transform = cv2.getPerspectiveTransform(source, corners)
    photo = cv2.warpPerspective(image, transform, (1200, 1000), borderValue=(180, 180, 180))
    return photo, corners


class GridSizeTests(unittest.TestCase):
    def test_varied_counts_perspective_and_lighting(self):
        for rows, cols in [(2, 2), (3, 8), (5, 6), (9, 4), (12, 15)]:
            for angled, shadow in [(False, False), (True, False), (True, True)]:
                with self.subTest(rows=rows, cols=cols, angled=angled, shadow=shadow):
                    frame, corners = grid_photo(rows, cols, angled, shadow)
                    self.assertEqual(detect_grid_size(frame, corners), (rows, cols))

    def test_blank_photo_does_not_default_to_five_by_six(self):
        frame = np.full((600, 800, 3), 255, np.uint8)
        with self.assertRaises(GridSizeError):
            detect_grid_size(frame, [[10, 10], [790, 10], [790, 590], [10, 590]])

    def test_missing_line_is_rejected(self):
        frame, corners = grid_photo(5, 6, missing_line=True)
        with self.assertRaises(GridSizeError):
            detect_grid_size(frame, corners)

    def test_invalid_corners_rejected(self):
        with self.assertRaises(GridSizeError):
            detect_grid_size(np.zeros((100, 100, 3), np.uint8), np.zeros((4, 2)))

    def test_detection_after_display_resize(self):
        frame, corners = grid_photo(9, 4, shadow=True)
        frame, corners = fit_photo(frame, corners, 800, 600)
        self.assertEqual(detect_grid_size(frame, corners), (9, 4))

    def test_photo_cli_uses_detected_counts(self):
        frame, corners = grid_photo(3, 8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "grid.png"
            output = Path(directory) / "overlay.png"
            ok, data = cv2.imencode(".png", frame)
            self.assertTrue(ok)
            data.tofile(path)
            with patch("examples.demo_ar.sample", wraps=sample) as fixture:
                main(["--image", str(path), "--auto-grid", "--corners",
                      *map(str, corners.ravel()), "--output", str(output)])
                self.assertEqual(fixture.call_args.args[:2], (3, 8))
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
