"""Tests for user-friendly UI behavior."""

import numpy as np

from ui import classify_image


def test_missing_image_returns_friendly_message() -> None:
    labels, conclusion, warning = classify_image(None)
    assert labels == {}
    assert conclusion == "Vui lòng chọn ảnh trước."
    assert warning == ""


def test_invalid_image_returns_friendly_message() -> None:
    invalid_image = np.empty((10, 10, 2), dtype=np.uint8)
    labels, conclusion, warning = classify_image(invalid_image)
    assert labels == {}
    assert conclusion.startswith("Không thể xử lý ảnh:")
    assert warning == ""
