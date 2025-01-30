# Facial Feature Extraction using Faster R-CNN

## Introduction
Facial feature extraction plays a crucial role in various applications such as facial recognition, emotion analysis, and augmented reality. This project focuses on utilizing a Faster R-CNN model to detect and extract key facial features from images. Faster R-CNN is a deep learning-based object detection framework known for its accuracy and efficiency.

This repository provides an end-to-end pipeline for training, evaluating, and testing a Faster R-CNN model using a custom dataset labeled in YOLO format. The model is fine-tuned on facial features and optimized for real-time applications.

## Proposed Methodology
The methodology consists of several key steps:

### 1. Data Preprocessing
- The dataset consists of facial images annotated in YOLO format.
- The labels are converted to the Faster R-CNN format (bounding box coordinates normalized to pixel values).
- Data augmentation techniques are applied to enhance model generalization.

### 2. Model Architecture
- We utilize the `fasterrcnn_resnet50_fpn` model from `torchvision.models.detection`.
- The classification head is modified to match the number of classes in our dataset.
- The model is initialized with pre-trained weights on COCO and fine-tuned on our dataset.

### 3. Training
- The training pipeline includes:
  - Optimizer: Stochastic Gradient Descent (SGD) with momentum
  - Loss Function: Multi-task loss (classification and bounding box regression)
  - Learning Rate Scheduler: StepLR to adjust learning rates dynamically
- The model is trained over multiple epochs, tracking losses and performance metrics.

### 4. Evaluation
- The trained model is evaluated using:
  - Precision, Recall, and F1-score for each class
  - Intersection over Union (IoU) to assess bounding box accuracy
- Metrics are logged and visualized for analysis.

### 5. Inference and Visualization
- The trained model is used for inference on test images.
- OpenCV is used to visualize the bounding boxes and detected features.
- The results are compared against ground-truth labels.

## Testing and Results
- The model was trained on a dataset of facial images with annotated features.
- After training, the following results were observed:
  - **Precision**: Achieved a high precision indicating fewer false positives.
  - **Recall**: Sufficient recall ensuring most features are detected.
  - **F1-score**: Balanced F1-score confirming model effectiveness.
- Qualitative results demonstrated accurate feature localization.

## Technologies Used
- Python 3.8+
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- Numpy

---
