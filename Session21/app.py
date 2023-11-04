import gradio as gr
import torch
from torch import nn
from torch.nn import functional as F
from nano_gpt_inferencing import generate_paragraph


HTML_TEMPLATE = """    
     <style>
        body {
            font-family: 'Arial', sans-serif;
            background: #3498db; /* Blue background color */
            margin: 0;
            padding: 0;
        }
        #app-header {
            text-align: center;
            background: rgba(255, 255, 255, 0.7); /* Semi-transparent white */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative; /* To position the artifacts */
            margin: 30px auto;
            max-width: 600px;
        }
        #app-header h1 {
            color: #2986cc;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header-images {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        .header-image {
            width: 100px;
            height: 100px;
            margin: 0 10px;
            background: #fff;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .header-image img {
            max-width: 80px;
            max-height: 80px;
            border-radius: 50%;
        }
        .concept-description {
            position: absolute;
            bottom: -30px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .concept:hover .concept-description {
            opacity: 1;
        }
    </style>
</head>
<body>
<div id="app-header">
    <!-- Header Images -->
    <div class="header-images">
        <div class="header-image">
            <img src="https://github.com/nkanungo/ERAS20/blob/main/images/bk.jpg?raw=true" alt="Image 1">
        </div>
        <div class="header-image">
            <img src="https://github.com/nkanungo/ERAS20/blob/main/images/sp.jpg?raw=true" alt="Image 2">
        </div>
    </div>
    <!-- Content -->
    <h1>Paragraph Auto Completion like Shakespeare </h1>
    <p>Generate dialogue using the intelligence from Shakespeare Dataset .</p>
    <p>Model: GPT.</p>
    <p>Dataset: Tiny Shakespeare.</p>
    <p>Token limit: User input .</p>
    <p>Input Text: User input.</p>
</div>
"""
with gr.Blocks(theme=gr.themes.Glass(),css=".gradio-container {background: url('https://github.com/nkanungo/ERAS20/blob/main/images/bg_1.jpg?raw=true')}") as interface:
    gr.HTML(value=HTML_TEMPLATE, show_label=False)
    
    with gr.Row(scale=1):
       
        
        inputs = [
            gr.Textbox(label="Input Text Prompt"),
            gr.Textbox(label="Token Limit")
        ]
        outputs = gr.Textbox(
            label="Generated Paragraph"
        )

     
    with gr.Column(scale=1):
        button = gr.Button("Generate")
        button.click(generate_paragraph, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    interface.launch(enable_queue=True)
