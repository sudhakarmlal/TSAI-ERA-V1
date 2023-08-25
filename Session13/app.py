import numpy as np
import gradio as gr
from src.detect import predict

gr.Interface(
    predict,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
        gr.Slider(0, 1, value=0.4, label="Threshold"),
        gr.Checkbox(label="Show Grad Cam"),
        gr.Slider(0, 1, value=0.5, label="Opacity of GradCAM"),
    ],
    outputs=gr.Gallery(rows=2, columns=1),
    title="YoloV3 on PASCAL VOC Dataset From Scratch (Slide for GradCam output)",
    examples=[
        ["example_images/009922.jpg", 0.5, 0.4, True, 0.5],
        ["example_images/009938.jpg", 0.6, 0.5, True, 0.5],
        ["example_images/009948.jpg", 0.55, 0.45, True, 0.5],
        ["example_images/009952.jpg", 0.5, 0.4, True, 0.5],
        ["example_images/009953.jpg", 0.6, 0.7, True, 0.5],
        ["example_images/009956.jpg", 0.5, 0.4, True, 0.5],
        ["example_images/009957.jpg", 0.6, 0.5, True, 0.5],
        ["example_images/009960.jpg", 0.55, 0.45, True, 0.5],
        ["example_images/009961.jpg", 0.5, 0.4, True, 0.5],
        ["example_images/009962.jpg", 0.6, 0.7, True, 0.5],
    ],
    layout="horizontal"
).launch()