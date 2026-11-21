"""Model loading and inference for the flower classifier."""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

from labels import CLASS_NAMES


IMAGE_SIZE = (180, 180)
MODEL_PATH = Path(__file__).resolve().parent / "flowers_cnn_model.h5"


class PredictionError(ValueError):
    """Raised when an input image cannot be processed."""


def _load_model() -> tf.keras.Model:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại {MODEL_PATH}. "
            "Hãy đặt flowers_cnn_model.h5 trong cùng thư mục với app.py."
        )

    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if loaded_model.input_shape != (None, 180, 180, 3):
        raise RuntimeError(
            "Mô hình không có đầu vào mong đợi (None, 180, 180, 3): "
            f"nhận được {loaded_model.input_shape}."
        )
    if loaded_model.output_shape != (None, len(CLASS_NAMES)):
        raise RuntimeError(
            f"Mô hình phải trả về {len(CLASS_NAMES)} logits: "
            f"nhận được {loaded_model.output_shape}."
        )
    return loaded_model


# Loaded once when the process starts, not once per prediction.
MODEL = _load_model()


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """Convert a Gradio RGB image to the model's BGR input batch."""
    if image_rgb is None:
        raise PredictionError("Vui lòng chọn ảnh trước.")

    image = np.asarray(image_rgb)
    if image.size == 0:
        raise PredictionError("Ảnh trống hoặc không thể đọc được.")
    if not np.issubdtype(image.dtype, np.number):
        raise PredictionError("Dữ liệu ảnh không hợp lệ.")
    if not np.all(np.isfinite(image)):
        raise PredictionError("Ảnh chứa dữ liệu không hợp lệ.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise PredictionError("Ảnh phải có 1, 3 hoặc 4 kênh màu.")

    # Gradio normally returns uint8. Also accept float images in [0, 1].
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image_bgr, IMAGE_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict(image_rgb: np.ndarray) -> Dict[str, float]:
    """Return a probability for each class in the model's label order."""
    batch = preprocess_image(image_rgb)
    logits = MODEL(batch, training=False)
    probabilities = tf.nn.softmax(logits[0]).numpy()
    return {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }
