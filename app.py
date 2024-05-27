import torch
import numpy as np
import gradio as gr
from transformers import GroundingDinoForObjectDetection, AutoProcessor

deviceTarget = "mps" if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu"
print(f"Using {deviceTarget} device")
device = torch.device(device=deviceTarget)
model = GroundingDinoForObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model.to(device);

def app_fn(
    image: gr.Image,
    labels: str,
    box_threshold: float,
    text_threhsold: float,
) -> str:
    labels = labels.split("\n")
    labels = [label if label.endswith(".") else label + "." for label in labels]
    labels = " ".join(labels)
    inputs = processor(images=image, text=labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threhsold,
        target_sizes=[image.size[::-1]]
    )[0]

    # convert tensor of [x0,y0,x1,y1] to list of [x0,y0,x1,y1] (int)
    boxes = result["boxes"].int().cpu().tolist()
    pred_labels = result["labels"]
    annot = [(tuple(box), pred_label) for box, pred_label in zip(boxes, pred_labels)]

    return (image, annot)

if __name__=="__main__":
    title = "Grounding DINO 🦖 for Object Detection"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            """
            This app demonstrates the use of the Grounding DINO model for object detection using the Hugging Face Transformers library.
            Grounding DINO is known for its strong ability of zero-shot object detection, thus it can detect objects in images based on textual descriptions.
            You can try the model by uploading an image and providing a textual description of the objects you want to detect or by splitting
            the description in different lines (this is how you can pass multiple labels). The model will then highlight the detected objects in the image 🤗
            """
        )
        with gr.Row():
            box_threshold = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.05, label="Box Threshold")
            text_threshold = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.05, label="Text Threshold")
            labels = gr.Textbox(lines=2, max_lines=5, label="Labels")
        btn = gr.Button()
        with gr.Row():
            img = gr.Image(type="pil")
            annotated_image = gr.AnnotatedImage()

        btn.click(fn=app_fn, inputs=[img, labels, box_threshold, text_threshold], outputs=[annotated_image])

    demo.launch()