#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import gradio as gr
from Era_s20_updt import generate_final_image


gr.Interface(
  
    generate_final_image,
    inputs=[
        #gr.Image(label="Input Image"),
        gr.Image(type='pil', label="Guided Image for Loss"),
        gr.Text(label="Input Prompt")
        
        #gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
        #gr.Slider(0, 1, value=0.4, label="Threshold"),
        #gr.Checkbox(label="Show Grad Cam"),
        #gr.Slider(0, 1, value=0.5, label="Opacity of GradCAM"),
    ],
    outputs =    
    [
        gr.Gallery(rows=2, columns=1"),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1),
        gr.Gallery(rows=2, columns=1)
        
    ],  
    title="Stable Diffusion",
    layout="Vertical"
).launch()

