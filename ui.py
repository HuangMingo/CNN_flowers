"""Gradio user interface for local flower recognition."""

from typing import Dict, Tuple

import gradio as gr
import numpy as np

from labels import VIETNAMESE_NAMES, display_name
from predictor import PredictionError, predict

LOW_CONFIDENCE_THRESHOLD = 0.50


def classify_image(
    image_rgb: np.ndarray,
) -> Tuple[Dict[str, float], str, str]:
    """Adapt predictor output to the three UI result components."""
    if image_rgb is None:
        return {}, "Vui lòng chọn ảnh trước.", ""

    try:
        probabilities = predict(image_rgb)
    except PredictionError as exc:
        return {}, f"Không thể xử lý ảnh: {exc}", ""
    except Exception:
        return {}, "Không thể đọc hoặc dự đoán ảnh này. Vui lòng thử ảnh khác.", ""

    best_class = max(probabilities, key=probabilities.get)
    best_probability = probabilities[best_class]
    label_values = {
        display_name(class_name): probability
        for class_name, probability in probabilities.items()
    }
    conclusion = (
        f"Kết luận: {VIETNAMESE_NAMES[best_class]} ({best_class}) — "
        f"{best_probability:.1%}"
    )
    warning = ""
    if best_probability < LOW_CONFIDENCE_THRESHOLD:
        warning = (
            "⚠️ Mô hình không chắc chắn, ảnh có thể không phải một trong "
            "5 loài hỗ trợ."
        )
    return label_values, conclusion, warning


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Nhận diện loài hoa") as demo:
        gr.Markdown(
            "# Nhận diện loài hoa\n"
            "Tải ảnh lên để mô hình CNN nhận diện loài hoa. "
            "Ảnh chỉ được xử lý cục bộ trên máy này."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="numpy",
                    sources=["upload", "clipboard"],
                    label="Ảnh cần nhận diện",
                )
                with gr.Row():
                    predict_button = gr.Button("Nhận diện", variant="primary")
                    clear_button = gr.Button("Xóa")

            with gr.Column():
                prediction_output = gr.Label(
                    num_top_classes=3,
                    label="Top 3 kết quả",
                )
                conclusion_output = gr.Textbox(
                    label="Kết luận",
                    interactive=False,
                )
                warning_output = gr.Markdown()

        # gr.Markdown(
        #     "**Hỗ trợ:** hoa hồng, hoa cúc, bồ công anh, "
        #     "hướng dương và tulip."
        # )
        # gr.Markdown(
        #     "*Mô hình chỉ phân loại trong 5 nhóm trên và có thể dự đoán sai. "
        #     "Kết quả có độ tin cậy dưới 50% sẽ được cảnh báo.*"
        # )

        predict_button.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[prediction_output, conclusion_output, warning_output],
        )
        clear_button.click(
            fn=lambda: (None, {}, "", ""),
            inputs=None,
            outputs=[
                image_input,
                prediction_output,
                conclusion_output,
                warning_output,
            ],
            queue=False,
        )

    return demo


demo = build_demo()
