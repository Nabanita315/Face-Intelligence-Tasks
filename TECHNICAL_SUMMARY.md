# Technical Summary: Face-Intelligence-Tasks

## Overview

The "Face-Intelligence-Tasks" repository offers solutions for two advanced computer vision tasks using the FACECOM dataset:

- **Task A: Gender Classification** — A binary classification for a facial image to be either a male, or a female.
- **Task B:  Face Recognition with Image Distortion — Multi-class classification to recognize distorted or edited images of faces to genuine faces.

---

## Approach & Architecture

### Task A: Gender Classification

- **Model**: It is based on the pre-trained ResNet-18 convolutional neural network, which is fine- tuned to predict gender.
- **Data Pipeline**:Dataset structure is under train/, val/ folder that separately has male/, female/ folder.
- **Training**:  Notebook (task-a.) orchestates data loading, augmentation, model training, validation, and testing. ipynb). Comparison of performance indicators are presented, including F1-score, confusion matrix, recall, accuracy, and precision.
- **Adaptability**: It is easy to set up paths for training, validation, and testing datasets, which makes it easy to add new datasets with few changes.

### Task B: Face Recognition with Distorted Images

- **Model**: Employs a pretrained **FaceNet** model to extract facial embeddings.
- **Matching Algorithm**: Compares embeddings of distorted images against clean reference images using **cosine similarity**.
- **Directory Structure**: Data is organized per person, with clean images in the root and distorted images under a `distortion/` subfolder.
- **Evaluation**: Generates matching statistics (accuracy, F1-score) and visualizations, including similarity histograms and threshold tuning plots.

---

## Innovations

- **Distortion Robustness**: A solution that can handle distorted images, a challenging scenario in face recognition, is specifically designed to address Task B.
- **Plug-and-Play Architecture**: Both tasks make use of tidy, well-documented Jupyter notebooks, making it easy to plug in extra models or transfer to different datasets.
- **Visualization**: Both exercises provide a visual representation of the results, which makes debugging and interpretation easier.
- **Adaptable Data Paths**: Since every data path is parameterized, adding new data can be done without requiring code rewriting.
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

Using the best pretrained models available, clever data organization, and practical engineering to facilitate rapid testing and adaptation, this repository demonstrates robust, modular approaches to challenging face intelligence tasks.
