"""Regression tests for preprocessing and model inference."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow as tf

from labels import CLASS_NAMES
from predictor import MODEL, predict, preprocess_image


SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_PATHS = sorted(SAMPLES_DIR.glob("*/*.jpg"))


def test_model_contract() -> None:
    assert MODEL.input_shape == (None, 180, 180, 3)
    assert MODEL.output_shape == (None, 5)


def test_rgb_preprocessing_matches_original_opencv_pipeline() -> None:
    if not SAMPLE_PATHS:
        pytest.skip("Không có ảnh mẫu trong tests/samples.")

    image_bgr = cv2.imread(str(SAMPLE_PATHS[0]))
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    actual = preprocess_image(image_rgb)
    expected = np.expand_dims(
        cv2.resize(image_bgr, (180, 180)).astype(np.float32) / 255.0,
        axis=0,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sample_path", SAMPLE_PATHS)
def test_predictor_matches_original_script(sample_path: Path) -> None:
    image_bgr = cv2.imread(str(sample_path))
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    actual = predict(image_rgb)
    original_batch = np.expand_dims(
        cv2.resize(image_bgr, (180, 180)).astype(np.float32) / 255.0,
        axis=0,
    )
    original_probabilities = tf.nn.softmax(
        MODEL(original_batch, training=False)[0]
    ).numpy()

    assert tuple(actual) == CLASS_NAMES
    np.testing.assert_allclose(
        list(actual.values()), original_probabilities, rtol=0, atol=1e-4
    )
    assert abs(sum(actual.values()) - 1.0) < 1e-5
