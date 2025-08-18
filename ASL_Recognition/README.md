
# Real-Time Sign Language Detection

## Overview

This project presents a pipeline for real-time sign language recognition using computer vision techniques and deep learning. Our aim is to bridge communication gaps for individuals with hearing impairments by leveraging a combination of from-scratch computer vision, deep neural networks, and keypoint-based landmark detection.

## Features

- End-to-end sign language recognition from video input
- Real-time gesture-to-sentence mapping
- Neural network-based hand detection and bounding box regression
- Integration of classical feature extraction and modern deep learning
- Comparison between multiple approaches (MediaPipe, SIFT, custom CNN)
- Dataset augmentation and preprocessing

---

## Dataset

### WL-ASL Dataset

- **Content**: 21083 videos of 2000 isolated signs (words & letters)
- **Distribution**:
  - Train: 14289 videos
  - Test: 2878 videos
  - Validation: 3916 videos
- **Augmentation**: Downloaded missing videos using custom Python scripts and Pytube; applied standard augmentations (flipping, rotation, scaling)

### EgoHands Dataset

- **Content**: 48000+ annotated images for hand detection
- **Usage**: Used a 1,000-sample subset for hand feature extraction

---

## Technical Details

### Hand Detection Model (from Scratch)

- **Feature Extraction**:
  - Canny & Sobel edge detection
  - Contour, convex hull, and convexity defect analysis
  - Local Binary Pattern (LBP) for texture
  - Skin color detection in HSV color space
  - Histogram of Oriented Gradients (HOG)
  - Standardization of features

- **Model Architecture**:
  - Shared Conv1D-based architecture with L2 regularization and dropout for robustness
  - Two outputs:
    - Binary classification (hand presence)
    - Regression (bounding box prediction)
  - Losses: Binary cross-entropy (classification), mean squared error (regression)
  - Trained with K-Fold cross validation

**Performance:**

| Model | Classification Precision | Classification Recall | Regression RMSE |
|-------|-------------------------|----------------------|-----------------|
| CNN   | 0.9925                  | 0.9968               | 0.1374          |

---

### Sign Detection Using MediaPipe

- **Frame Extraction**: 5 fps sampling, stored according to label
- **Hand and Landmark Detection**: MediaPipe with 21 landmark points per hand; average hand bounding box accuracy ~95.7%
- **Model**: Fully-Connected Neural Network (FCNN) with batch normalization and dropout
- **Label Input**: All 21 landmarks per hand, flattened; softmax final layer for multi-class classification

**Performance:**

- Training Accuracy: 87%
- Validation Accuracy: 61%

---

### Sign Detection Using SIFT

- **Feature Extraction**: SIFT descriptors extracted from hand detection; vectors padded and flattened
- **Model**: FCNN with more neurons (due to high-dimensional SIFT input)
- **Limitations**: Lower accuracy due to inconsistent keypoints and noise

**Performance:**

- Accuracy: 48%
- Loss: 2.91

---

## Real-Time Gesture-to-Sentence Mapping

- Classified gestures are mapped in sequence to output sentences (using OpenCV & custom mapping)
- Ensures generated output maintains temporal and grammatical correctness

---

## Approaches Compared

| Method      | Strengths                        | Limitations                     |
|-------------|----------------------------------|---------------------------------|
| CNN (scratch) | High hand detection/classification accuracy | Regression/bounding-box RMSE can improve |
| MediaPipe   | Robust, high accuracy with labeled keypoints | Moderate validation accuracy, potential improvement via hyperparameters |
| SIFT        | Rotation/scale invariance        | Poor generalizability, noise sensitivity |

---

## Improvements & Future Work

- Enhance regression performance with more annotated data and architecture improvements (e.g., attention mechanisms)
- Fine-tune keypoint confidence and frame rates in MediaPipe approach
- Potential switch to recurrent models (RNN/LSTM) for temporal memory
- More data cleaning (background removal, frame selection, etc.)

---
