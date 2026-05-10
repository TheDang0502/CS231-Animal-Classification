"""
Gradio Web UI for Animal Image Classification
Using VGG16 + Random Forest
"""

import gradio as gr
from pathlib import Path
from src.predict import AnimalClassifier

import os
import cv2
import tempfile

# =========================
# MODULE PATH
# =========================
MODULES_PATH = (
    Path(__file__).parent / 'modules'
).resolve()

classifier = None


def load_classifier():
    """Load classifier once."""

    global classifier

    if classifier is None:

        print("Loading classifier...")

        classifier = AnimalClassifier(
            str(MODULES_PATH)
        )


def predict_animal(image):
    """
    Predict animal from uploaded image
    """

    if image is None:
        return {}

    load_classifier()

    # RGB → BGR
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Save temp image
    with tempfile.NamedTemporaryFile(
        suffix='.jpg',
        delete=False
    ) as f:

        temp_path = f.name

        cv2.imwrite(
            temp_path,
            image_bgr
        )

    try:

        result = classifier.predict(
            temp_path
        )

        # Return probabilities
        if "probabilities" in result:

            return result["probabilities"]

        return {
            result["predicted_class"]: 1.0
        }

    except Exception as e:

        return {
            "error": str(e)
        }

    finally:

        if os.path.exists(temp_path):

            os.unlink(temp_path)


# =========================
# UI
# =========================
with gr.Blocks(
    title="Animal Image Classifier"
) as demo:

    gr.Markdown(
        """
        # 🐾 Animal Image Classifier

        Upload an image to classify animals using:

        **VGG16 + Random Forest + PCA**
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                label="Upload Image",
                type="numpy"
            )

            predict_btn = gr.Button(
                "Predict",
                variant="primary"
            )

        with gr.Column():

            output_label = gr.Label(
                num_top_classes=5
            )

    predict_btn.click(
        fn=predict_animal,
        inputs=image_input,
        outputs=output_label
    )


if __name__ == "__main__":

    print("Loading model...")

    load_classifier()

    print("Starting Gradio app...")

    demo.launch()