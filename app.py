"""Start the local Gradio flower-recognition application."""

from ui import demo

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=3000,
        share=False,
        inbrowser=True,
    )
