# Gradio-GroundingDINO-Mac
This repository hosts a Gradio application that leverages PyTorch and the transformers library to perform zero-shot object detection using the grounding-dino-tiny model from IDEA-Research. The application is enabled to run on Mac silicon chips, ensuring efficient performance on Apple silicon chips. Also can run with CPU or CUDA!

## Features:
Zero-Shot Object Detection: Utilize the state-of-the-art grounding-dino-tiny model to detect objects without the need for prior training on specific classes.
Gradio Interface: A user-friendly web interface for uploading images and viewing detection results in real-time.
Mac Silicon Optimization: Fully compatible and optimized for running on Mac silicon chips (M1, M2), leveraging their computational capabilities.
Easy Setup: Simple installation and setup process with detailed instructions to get you started quickly.
Installation:
Follow these steps to set up the application on your local machine:

# Clone the Repository:

```
git clone https://github.com/Aurorabyte-Solutions/Gradio-GroundingDINO-Mac.git
cd Gradio-GroundingDINO-Mac

```

# Create and Activate a Virtual Environment:
```
python3 -m venv venv
source venv/bin/activate

```

# Install Dependencies:
```
pip install -r requirements.txt
```

# Run the Gradio App:
```
python app.py
```

# Usage:
Open your web browser and navigate to http://localhost:7860.
Upload an image or provide a URL to an image.
Enter the object classes you want to detect (comma-separated).
Click the "Submit" button to view the detection results.

# Example:
Here’s a quick example to illustrate the usage:

Upload an Image:

View Detection Results:

<img width="1483" alt="image" src="https://github.com/Aurorabyte-Solutions/Gradio-GroundingDINO-Mac/assets/25804457/66777cf2-8ed2-4f74-8316-fadef8eed1b7">

### Requirements:
Python 3.7 or higher
Any computer, CPU, CUDA OR MPS (for mac sillicon chips)


### Contributing:
Contributions are welcome! Please fork this repository and submit a pull request with your changes.

### License:
This project is licensed under the MIT License.

### Acknowledgments:
Thanks to IDEA-Research for the grounding-dino-tiny model.
Based on the work of  EduardoPacheco HuggingFace [space](https://huggingface.co/spaces/EduardoPacheco/Grounding-Dino-Inference)
