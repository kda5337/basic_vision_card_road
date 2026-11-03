from pathlib import Path

import cv2
import numpy as np

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def preprocess(frame):
    enhanced = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def find_card_candidates(frame, edges):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = frame.shape[0] * frame.shape[1] * 0.001
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            points = polygon.reshape(4, 2)
            w, h = cv2.minAreaRect(polygon)[1]

        else:
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            points = cv2.boxPoints(rect).astype(np.int32)

        if w <= 0 or h <= 0:
            continue

        ratio = max(w, h) / min(w, h)
        rectangularity = area / (w * h)

        if ratio < 1.05 or ratio > 1.8 or rectangularity < 0.65:
            continue

        candidates.append((area, points))

    cards = []

    candidates.sort(key=lambda item: item[0], reverse=True)

    for area, points in candidates:
        x, y, w, h = cv2.boundingRect(points)
        duplicate = False

        for _, old_points in cards:
            ox, oy, ow, oh = cv2.boundingRect(old_points)

            ix1, iy1 = max(x, ox), max(y, oy)
            ix2, iy2 = min(x + w, ox + ow), min(y + h, oy + oh)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = w * h + ow * oh - intersection

            if union > 0:
                iou = intersection / union

                if iou > 0.60:
                    duplicate = True
                    break

        if not duplicate:
            cards.append((area, points))

    return sorted(
        [points for _, points in cards],
        key=lambda points: np.asarray(points).reshape(4, 2)[:, 0].mean(),
    )


def order_points(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0)


def warp_card(frame, points):
    tl, tr, br, bl = order_points(points)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if min(width, height) < 1:
        return None, None

    scale = min(1.0, 600 / max(width, height))
    out_w = max(2, int(width * scale))
    out_h = max(2, int(height * scale))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        order_points(points), destination
    )
    card = cv2.warpPerspective(frame, matrix, (out_w, out_h))
    return card, np.linalg.inv(matrix)


def load_templates(templates_dir=TEMPLATES_DIR):
    """Load grayscale symbol images, keyed by filename without extension."""
    templates = {}
    for path in sorted(Path(templates_dir).glob("*.png")):
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates[path.stem] = template
    return templates


def _extract_symbol(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_area = max(cv2.contourArea(c) for c in contours)
    symbol = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) >= max(4, largest_area * 0.12):
            cv2.drawContours(symbol, [contour], -1, 255, -1)

    locations = cv2.findNonZero(symbol)
    if locations is None:
        return None

    x, y, width, height = cv2.boundingRect(locations)
    return symbol[y:y + height, x:x + width]


def _normalize_symbol(symbol, size=64, margin=6):
    if symbol is None or symbol.size == 0:
        return None
    height, width = symbol.shape[:2]
    scale = min((size - 2 * margin) / width, (size - 2 * margin) / height)
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        symbol, (out_width, out_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - out_width) // 2
    y = (size - out_height) // 2
    canvas[y:y + out_height, x:x + out_width] = resized
    return canvas


def _symbol_similarity(image_a, image_b):
    correlation = cv2.matchTemplate(image_a, image_b, cv2.TM_CCOEFF_NORMED)[0, 0]

    mask = np.ones((3, 3), dtype=np.uint8)
    image_a = cv2.dilate(image_a, mask)
    image_b = cv2.dilate(image_b, mask)

    intersection = np.count_nonzero((image_a > 0) & (image_b > 0))
    union = np.count_nonzero((image_a > 0) | (image_b > 0))
    iou = intersection / union if union else 0

    similarity = 0.5 * ((correlation + 1) / 2) + 0.5 * iou
    return float(similarity)


def detect_arrow(
    card,
    templates=None,
    debug=False,
):
    """기호 전체를 정규화해 템플릿과 비교한다.

    Returns:
        가장 유사한 템플릿 라벨과 유사도 점수.
    """
    if card is None or card.size == 0:
        return "UNKNOWN", 0.0
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    if templates is None:
        templates = load_templates()
    if not templates:
        return "UNKNOWN", 0.0

    height, width = gray.shape[:2]
    roi = gray[int(height * 0.15):int(height * 0.65), int(width * 0.15):int(width * 0.85)]
    if roi.size == 0:
        return "UNKNOWN", 0.0

    card_symbol = _normalize_symbol(_extract_symbol(cv2.GaussianBlur(roi, (3, 3), 0)))
    if card_symbol is None:
        return "UNKNOWN", 0.0

    scores = {}
    for label, template in templates.items():
        template_gray = template
        template_symbol = _normalize_symbol(_extract_symbol(template_gray))
        if template_symbol is None:
            continue
        scores[label] = _symbol_similarity(card_symbol, template_symbol)

    if not scores:
        return "UNKNOWN", 0.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]

    if debug:
        print("Template scores:", ", ".join(
            f"{label}={score:.3f}" for label, score in ranked
        ))

    return best_label, float(best_score)

def analyze_frame(frame, templates=None):
    edges = preprocess(frame)
    cards = find_card_candidates(frame, edges)
    if templates is None:
        templates = load_templates()

    card_results = {}
    for card_id, points in enumerate(cards, start=1):
        card, _ = warp_card(frame, points)
        label, _ = detect_arrow(card, templates) if card is not None else ("UNKNOWN", 0.0)
        card_results[card_id] = label if label is not None else "UNKNOWN"

    return cards, card_results


_COMMAND_MEMBER_NAMES = {
    "forward": ("FORWARD", "MOVE_FORWARD"),
    "left_double": ("LOOP_END",),
    "multiply_2": ("MULTIPLY_2", "REPEAT_2", "LOOP_2"),
    "multiply_3": ("MULTIPLY_3", "REPEAT_3", "LOOP_3"),
    "rotate_left": ("TURN_LEFT", "ROTATE_LEFT"),
    "rotate_right": ("TURN_RIGHT", "ROTATE_RIGHT"),
}


def _get_command(label, Command):
    for member_name in _COMMAND_MEMBER_NAMES.get(label, ()):
        command = getattr(Command, member_name, None)
        if command is not None:
            return command
    return None


def detect_cards(frame):
    """웹캠 프레임 한 장에서 카드를 인식해 공통 결과 형식으로 반환한다.

    반환되는 CardData 목록은 위쪽 행부터, 각 행에서 왼쪽에서 오른쪽 순서다.
    카드가 없거나 기호를 판별/매핑할 수 없으면 data=None과 오류 코드를 반환한다.
    """
    from common.models import CardData, Command, DetectionResult

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        return DetectionResult(data=None, error_code="INVALID_FRAME")

    edges = preprocess(frame)
    card_points = find_card_candidates(frame, edges)
    if not card_points:
        return DetectionResult(data=None, error_code="NO_CARDS")

    templates = load_templates()
    if not templates:
        return DetectionResult(data=None, error_code="TEMPLATES_NOT_FOUND")
    if not set(_COMMAND_MEMBER_NAMES).issubset(templates):
        return DetectionResult(data=None, error_code="TEMPLATES_INCOMPLETE")

    cards = []
    for points in card_points:
        card, _ = warp_card(frame, points)
        if card is None:
            return DetectionResult(data=None, error_code="INVALID_CARD_LAYOUT")

        label, confidence = detect_arrow(card, templates)
        if label == "UNKNOWN":
            return DetectionResult(data=None, error_code="UNKNOWN_CARD")

        command = _get_command(label, Command)
        if command is None:
            return DetectionResult(data=None, error_code="COMMAND_MAPPING_UNAVAILABLE")

        bbox = cv2.boundingRect(points)
        cards.append(CardData(command=command, bbox=bbox, confidence=confidence))

    return DetectionResult(data=cards, error_code=None)

