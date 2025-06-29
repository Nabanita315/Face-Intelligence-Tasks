# Technical Summary: Face-Intelligence-Tasks

## Overview

The "Face-Intelligence-Tasks" repository provides solutions for two advanced computer vision challenges using the FACECOM dataset:

- **Task A: Gender Classification** — Binary classification of facial images as male or female.
- **Task B: Face Recognition with Distorted Images** — Multi-class classification to match distorted or altered face images to their correct identities.

---

## Approach & Architecture

### Task A: Gender Classification

- **Model**: Utilizes a pretrained ResNet-18 convolutional neural network, fine-tuned for gender classification.
- **Data Pipeline**: Dataset is organized into `train/` and `val/` folders, each containing `male/` and `female/` subfolders.
- **Training**: The notebook (`task-a.ipynb`) handles data loading, augmentation, model training, validation, and testing. Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix are reported.
- **Adaptability**: Paths for training, validation, and testing datasets are easily configurable, supporting new datasets with minimal changes.

### Task B: Face Recognition with Distorted Images

- **Model**: Employs a pretrained **FaceNet** model to extract facial embeddings.
- **Matching Algorithm**: Compares embeddings of distorted images against clean reference images using **cosine similarity**.
- **Directory Structure**: Data is organized per person, with clean images in the root and distorted images under a `distortion/` subfolder.
- **Evaluation**: Generates matching statistics (accuracy, F1-score) and visualizations, including similarity histograms and threshold tuning plots.

---

## Innovations

- **Distortion Robustness**: The solution for Task B is specifically designed to handle distorted images, a challenging scenario in face recognition.
- **Plug-and-Play Architecture**: Both tasks use clean, well-commented Jupyter notebooks, making it easy to adapt to other datasets or integrate further models.
- **Visualization**: Both tasks provide visualization of results, aiding interpretability and debugging.
- **Flexible Data Paths**: All data paths are parameterized, allowing seamless adaptation for new data without code rewrites.

---

## Requirements

- Python 3.8+
- torch, torchvision, facenet-pytorch
- numpy, pillow, matplotlib, scikit-learn, tqdm, seaborn

---

## Folder Structure

```
Comys_Hackathon5/
├── Task_A/
│   ├── train/   # Training images (male/, female/)
│   ├── val/     # Validation images (male/, female/)
│   └── task-a.ipynb
├── Task_B/
│   ├── train/   # Training images (person folders, distortion/)
│   ├── val/     # Validation images (person folders, distortion/)
│   └── task-b.ipynb
└── README.md
```

---

## Conclusion

This repository demonstrates robust, modular approaches to challenging face intelligence tasks, combining state-of-the-art pretrained models, thoughtful data organization, and practical engineering for rapid adaptation and evaluation.