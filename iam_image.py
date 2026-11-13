"""Image normalization, deskewing, and horizontal line segmentation."""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate
from scipy.signal import find_peaks



def rectify_page(image: Image.Image) -> Image.Image:
    """Perspective-correct a photographed page when a clear page quadrilateral exists."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(polygon)
        if len(polygon) == 4 and area >= height * width * 0.22 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2)))
    if not candidates:
        return image.convert("L")
    points = candidates[0][1].astype("float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered = np.array([
        points[np.argmin(sums)], points[np.argmin(diffs)],
        points[np.argmax(sums)], points[np.argmax(diffs)]
    ], dtype="float32")
    tl, tr, br, bl = ordered
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < width * 0.35 or out_h < height * 0.35 or out_w < 100 or out_h < 100:
        return image.convert("L")
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(gray, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Image.fromarray(warped)


def remove_notebook_lines(image: Image.Image) -> Image.Image:
    """Suppress long horizontal paper rules while trying to preserve ink crossings."""
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    if width < 80 or height < 40:
        return image.convert("L")
    block = max(31, min(61, (min(height, width) // 2) * 2 + 1))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 12)
    kernel_width = max(25, width // 28)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    line_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    # Ignore short horizontal strokes; only remove lines spanning much of the page.
    if cv2.countNonZero(line_mask) < width * 0.12:
        return image.convert("L")
    line_mask = cv2.dilate(line_mask, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = cv2.inpaint(gray, line_mask, 2, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)


def to_grayscale_array(image: Image.Image) -> np.ndarray:
    """Convert to grayscale and normalize polarity to dark ink on light paper."""
    arr = np.asarray(ImageOps.autocontrast(image.convert("L")), dtype=np.uint8)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    if np.median(border) < 127:
        arr = 255 - arr
    return arr


def estimate_skew_angle(image: Image.Image, max_angle: float = 7.0) -> float:
    """Estimate small page skew by maximizing horizontal projection variance."""
    arr = to_grayscale_array(image)
    if arr.size == 0:
        return 0.0
    background = gaussian_filter(arr.astype(np.float32), sigma=max(5, min(arr.shape) / 40))
    ink = np.clip(background - arr.astype(np.float32) - 8, 0, 255)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 0.001, 0.25):
        rotated = rotate(ink, float(angle), reshape=False, order=1, mode="constant", cval=0)
        projection = rotated.sum(axis=1)
        score = float(np.var(gaussian_filter1d(projection, sigma=1.5)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew_image(image: Image.Image, max_angle: float = 7.0) -> Image.Image:
    """Deskew a scanned page while preserving the full rotated canvas."""
    angle = estimate_skew_angle(image, max_angle=max_angle)
    if abs(angle) < 0.15:
        return image.convert("L")
    arr = to_grayscale_array(image)
    corrected = rotate(arr, angle, reshape=True, order=1, mode="constant", cval=255)
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))


def split_page_into_lines(image: Image.Image, min_line_height: int = 6) -> list[Image.Image]:
    """Deskew a page, find text bands from adaptive row projections, and crop lines.

    The adaptive threshold follows local background contrast, while small row gaps
    are closed to keep punctuation and ascenders/descenders attached to their line.
    """
    corrected = deskew_image(image)
    corrected = remove_notebook_lines(corrected)
    arr = to_grayscale_array(corrected)
    gray = arr.astype(np.float32)
    background = gaussian_filter(gray, sigma=max(4.0, min(arr.shape) / 35.0))
    ink = gray < (background - 10.0)
    if not ink.any():
        return []

    projection = gaussian_filter1d(ink.sum(axis=1).astype(np.float32), sigma=1.4)
    threshold = max(1.0, arr.shape[1] * 0.001)
    active = projection >= threshold
    # Bridge tiny gaps within a text line but retain real inter-line whitespace.
    gap_limit = max(4, min(12, int(arr.shape[0] * 0.012)))
    active_rows = np.flatnonzero(active)
    if not len(active_rows):
        return []

    groups = []
    first = previous = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row - previous > gap_limit + 1:
            if previous - first + 1 >= min_line_height:
                groups.append((first, previous + 1))
            first = row
        previous = row
    if previous - first + 1 >= min_line_height:
        groups.append((first, previous + 1))

    # If ink bands merge, split at clear projection valleys between line peaks.
    if len(groups) <= 1:
        peaks, _ = find_peaks(
            projection,
            distance=max(16, int(arr.shape[0] * 0.05)),
            prominence=max(2.0, float(projection.max()) * 0.025),
        )
        if len(peaks) >= 2:
            cuts = [
                int(left + np.argmin(projection[left:right + 1]))
                for left, right in zip(peaks[:-1], peaks[1:])
                if right - left >= 12
            ]
            candidate = []
            bounds = [0] + cuts + [arr.shape[0]]
            for top, bottom in zip(bounds[:-1], bounds[1:]):
                rows = np.flatnonzero(projection[top:bottom] >= threshold)
                if len(rows) and bottom - top >= min_line_height:
                    candidate.append((top + int(rows[0]), top + int(rows[-1]) + 1))
            if len(candidate) > len(groups):
                groups = candidate

    crops = []
    pad_y = max(3, int(arr.shape[0] * 0.004))
    pad_x = max(4, int(arr.shape[1] * 0.004))
    for y0, y1 in groups:
        region = ink[y0:y1]
        cols = np.flatnonzero(region.any(axis=0))
        if not len(cols):
            continue
        top, bottom = max(0, y0 - pad_y), min(arr.shape[0], y1 + pad_y)
        left, right = max(0, int(cols[0]) - pad_x), min(arr.shape[1], int(cols[-1]) + 1 + pad_x)
        if bottom - top >= min_line_height and right - left >= 12:
            crops.append(Image.fromarray(arr[top:bottom, left:right]))
    return crops
