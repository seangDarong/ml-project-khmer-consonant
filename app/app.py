"""Demo app: draw a Khmer consonant (or upload a photo) and see the model's top 3 guesses.

Run from the project root:
    python app/app.py
then open the link it prints (http://127.0.0.1:7860).

Needs models/cnn.pt, made by notebooks/02_train.ipynb.
"""
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import predict  # noqa: E402

LOW_CONFIDENCE = 0.40   # below this we tell the user we're not sure


def to_array(value):
    """Gradio gives the sketchpad as a dict with a 'composite' image, and an upload as a plain array."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
    return None if value is None else np.asarray(value)


def classify(drawing, photo):
    img = to_array(drawing)
    if img is None or (img.ndim == 3 and img.shape[2] == 4 and img[:, :, 3].max() == 0):
        img = to_array(photo)  # nothing drawn: use the uploaded photo instead
    if img is None:
        return {}, "Draw a letter or upload a photo first."

    top = predict(img, top_k=3)
    scores = {f"{t['letter']}  ({t['romanization']})": t["confidence"] for t in top}
    best = top[0]
    if best["confidence"] < LOW_CONFIDENCE:
        msg = f"Not sure. Best guess **{best['letter']}** ({best['romanization']}) at {best['confidence']:.0%}. Try drawing it bigger and clearer."
    else:
        msg = f"# {best['letter']}\n**{best['romanization']}**, {best['confidence']:.0%} confident"
    return scores, msg


with gr.Blocks(title="Khmer Consonant Recognizer") as demo:
    gr.Markdown("## Khmer Consonant Recognizer\nDraw one of the 33 Khmer consonants (ក … អ), large and centred, then press **Predict**.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                drawing = gr.Sketchpad(
                    canvas_size=(400, 400), type="numpy", image_mode="RGBA", label="Draw here",
                    brush=gr.Brush(default_size=14, colors=["#000000"], color_mode="fixed"),
                )
            with gr.Tab("Upload photo"):
                photo = gr.Image(type="numpy", label="Photo of one handwritten letter (dark ink on light paper)")
            button = gr.Button("Predict", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            top3 = gr.Label(num_top_classes=3, label="Top 3")
    button.click(classify, inputs=[drawing, photo], outputs=[top3, result])

if __name__ == "__main__":
    demo.launch()
