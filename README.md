# YOLO11n Bounding Box Annotation Noise — Interactive Demo

An interactive computer vision research demo comparing four YOLO11n object detection models trained with different levels of bounding-box annotation noise.

## Live Demo

https://yolo11n-public-demo-12.streamlit.app/

## Research Overview

This demo accompanies an independent research study:

**Impact of Bounding Box Annotation Noise on Object Detection Performance**

The study investigates how controlled changes to training bounding boxes affect object detection performance.

## Models

| Model | Training Annotation Noise |
|-------|---------------------------|
| YOLO11n | 0% |
| YOLO11n | 5% |
| YOLO11n | 10% |
| YOLO11n | 20% |

The demo allows users to compare predictions from the four trained models on the same input image.

## Technology

- Python
- Ultralytics YOLO11n
- PyTorch
- Streamlit
- OpenCV

## Run Locally

Install dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py

## Research Repository

https://github.com/tanvirshadgit/bounding-box-noise-research

## Research Notes

This application is intended for qualitative demonstration.

Predictions on individual images should not be interpreted as evidence of statistically significant differences between models.

For quantitative evaluation, experimental methodology, and independent test results, refer to the research repository.

## Author

Tanvir Ahmed Khan

Independent Computer Vision Researcher
