# ERA_S21


GPT-Based Paragraph Generation
================================
This Python code implements a paragraph generation model based on the GPT architecture, fine-tuned on the Tiny Shakespeare dataset. The model takes an initial text prompt and generates a coherent paragraph of text as output.

Prerequisites
==============
Python 3.x
PyTorch
Gradio (for the user interface)

Getting Started
==================
Clone or download this repository.
Install the required dependencies using pip install torch gradio.

Usage
============
Run the Python script to start the paragraph generation interface.
Visit the interface in your web browser

  https://huggingface.co/spaces/nkanungo/tinygpt


Input your text prompt in the "Input Text Prompt" field.
Specify the maximum token limit for the generated paragraph in the "Token Limit" field.
Click the "Generate" button to receive a generated paragraph.

Model Details
================
Model: GPT-based model.
Dataset: Tiny Shakespeare.
Project Structure
nano_gpt_model.pth: pretrained model weights.
nano_gpt_inferencing.py: Functions for generating paragraphs.
README.md: This README file.

Example
==============
For example, if you input "Once upon a time" as the text prompt and set the token limit to 50, the model will generate a paragraph based on this input.

Acknowledgments
=====================
The model architecture is based on the GPT architecture.
The Tiny Shakespeare dataset was used for fine-tuning.
License
This project is licensed under Apache 2.0.

Authors
=============
Manjunath Yellipeta
Nihar Ranjan Kanungo
