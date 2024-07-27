# Face Analysis

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-analysis/total) [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-analysis)

<video controls autoplay loop src="https://github.com/user-attachments/assets/d4bf1ed3-4f53-44ab-80ee-82e0df4d95e6
" muted="false" width="100%"></video>

This repository contains functionalities for face detection, age and gender classification, face recognition, and facial landmark detection. It supports inference from an image or webcam/video sources.

## Features

- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection.
- [x] **Gender & Age Classification**: Provides discrete age predictions and binary gender classification (male/female).
- [ ] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition.
- [ ] **Facial Landmark Detection**
- [x] **Real-Time Inference**: Supports both webcam and video file input for real-time processing.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakhyo/face-analysis.git
cd face-analysis
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakyo/face-analysis.git
cd face-analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download weight files:

   a) Download weights from the following links:

   | Model           | Weights                                                                                           |
   | --------------- | ------------------------------------------------------------------------------------------------- |
   | ArcFace         | [w600k_r50.onnx](https://github.com/yakhyo/face-analysis/releases/download/v0.0.1/w600k_r50.onnx) |
   | SCRFD           | [det_2.5g.onnx](https://github.com/yakhyo/face-analysis/releases/download/v0.0.1/det_2.5g.onnx)   |
   | SCRFD           | [det_500m.onnx](https://github.com/yakhyo/face-analysis/releases/download/v0.0.1/det_500m.onnx)   |
   | SCRFD (default) | [det_10g.onnx](https://github.com/yakhyo/face-analysis/releases/download/v0.0.1/det_10g.onnx)     |
   | GenderAge       | [genderage.onnx](https://github.com/yakhyo/face-analysis/releases/download/v0.0.1/genderage.onnx) |

   b) Run the command below to download weights to the `weights` directory (Linux):

   ```bash
   sh download.sh
   ```

## Usage

```bash
python main.py --source assets/in_video.mp4
                        assets/in_image.jpg
                        0 # for webcam
```

`main.py` arguments:

```
usage: main.py [-h] [--detection-weights DETECTION_WEIGHTS] [--attribute-weights ATTRIBUTE_WEIGHTS] [--source SOURCE] [--output OUTPUT]

Run face detection on an image or video

options:
  -h, --help            show this help message and exit
  --detection-weights DETECTION_WEIGHTS
                        Path to the detection model weights file
  --attribute-weights ATTRIBUTE_WEIGHTS
                        Path to the attribute model weights file
  --source SOURCE       Path to the input image or video file or camera index (0, 1, ...)
  --output OUTPUT       Path to save the output image or video
```

## Reference

1. https://github.com/deepinsight/insightface
