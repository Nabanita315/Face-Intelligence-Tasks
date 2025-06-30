# Comys Hackathon 5: Face-Intelligence-Tasks

This repository contains solutions for two challenging computer vision problems using the FACECOM
dataset:

- **Task A: Gender Classification (Binary Classification)**
- **Task B: Face Recognition with Distorted Images (Multi-Class Classification)**

---

## 📁 Folder Structure

```
Comys_Hackathon5/
│
├── Task_A/
│   ├── train/         # Training images (male/, female/)
│   ├── val/           # Validation images (male/, female/)
│   └── task-a.ipynb   # Gender classification notebook
│
├── Task_B/
│   ├── train/         # Training images (per-person folders)
│   ├── val/           # Validation images (per-person folders, with distortion/)
│   └── task-b.ipynb   # Face recognition notebook
│
└── README.md               # This file
```

---

## Task A: Gender Classification

**Objective:**  
Train a model to classify face images as either male or female.

**Highlights:**  
- Uses a pretrained ResNet18 model
- Includes data augmentation
- Handles class imbalance using weighted sampling.
- Evaluates performance with  **accuracy, precision, recall, F1-score** and **confusion matrix**.

**Workflow:**

**Flowchart:**

https://github.com/Nabanita315/Face-Intelligence-Tasks/blob/main/TaskA-Flowchart.png

**How to Run:**
1. Place your data in `Task_A/train/` and `Task_A/val/` with subfolders `male/` and `female/`.
2. Open a terminal or Jupyter/VS Code and **change your working directory to `Task_A`**:
    ```bash
    cd Comys_Hackathon5/Task_A
    ```
3. Open and run `task-a.ipynb`.
4. Ensure your dataset is placed in the following structure:
    ```
    Task_A/
      ├── train/
      │   ├── male/
      │   └── female/
      ├── val/
      │   ├── male/
      │   └── female/
      └── task-a.ipynb
    ```
5. The default dataset paths (`./train`, `./val`) will work as long as you run the notebook from inside the `Task_A` folder.
6. The "train_dir" and "val_dir" contains training data directory path and validation data directory path respectively.
7. For evaluating on a new dataset, change the path of "**test_dir**" with your data directory path.

---

## Task B: Face Recognition with Distorted Images

**Objective:**  
Match face images (including distorted versions) to their corresponding identities.

**Highlights:**
- Uses a pretrained **FaceNet** model for embedding extraction.
- Matches distorted images to clean ones using **cosine similarity**.
- Evaluates with **accuracy, F1 score**.
- Provides ** visualizations** of matching results.

**Workflow:**

This workflow describes each major step of the face recognition pipeline using Facenet-PyTorch, including data handling, feature extraction, matching, and evaluation.

---

1. **Setup & Dependencies**
- **Install required packages:**  
  `facenet-pytorch`, `torch`, `torchvision`, `pillow`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `tqdm`
- **Import libraries.**
- **Set device:**  
  Use GPU if available; fallback to CPU.

---

2. **Data and Model Preparation**
- **Define directory structure:**  
  - `train_dir` (for registration, if needed)  
  - `val_dir` (validation/test set, including "distortion" folders)
- **Load pretrained models:**  
  - `InceptionResnetV1` (FaceNet, for embeddings)  
  - `MTCNN` (for face detection and alignment)

---

3. **Embedding Extraction with TTA**
- **Test-Time Augmentation (TTA):**  
  For each image, generate multiple augmented versions (brightness, sharpness, contrast, original).
- **Face detection & alignment:**  
  Apply MTCNN to each augmented image.
- **Get embeddings:**  
  Pass detected faces through FaceNet and average the results to obtain a robust embedding.

---

4. **Build Validation Feature Database**
- **Iterate over each person in `val_dir`:**  
  - For each clean (non-distorted) image:
    - Extract embedding (using TTA).
    - Store embeddings in a dictionary:  
      `{person_name: [embedding1, embedding2, ...]}`
      
---

5. **Distorted Image Matching**
- **Iterate over each person’s "distortion" folder in `val_dir`:**  
  - For each distorted image:
    - Extract embedding.
    - Compute cosine similarity with every stored embedding in the validation feature DB.
    - Identify the most similar person (highest similarity score).

---

6. **Thresholding and Result Storage**
- **Apply similarity threshold:**  
  Only accept a match if the score exceeds a set threshold (e.g., 0.65).
- **Store results:**  
  Save file name, true label, predicted label, similarity score, and match indicator.

---

7. **Evaluation and Analysis**
- **Print sample results.**
- **Compute metrics:**  
  - Top-1 Accuracy  
  - Macro-averaged F1 Score
- **Threshold tuning:**  
  Plot accuracy and F1 as functions of the similarity threshold.
- **Cosine similarity histogram:**  
  Visualize the distribution of similarity scores for matches and mismatches.

---

8. **Visualization**
- **Threshold tuning curve:**  
  Shows how accuracy and F1 score vary with threshold.
- **Histogram of similarity scores:**  
  Compare genuine vs imposter match distributions.

---

9. **End**

---

**Summary Table**

| Step | Description                                              |
|------|----------------------------------------------------------|
| 1    | Setup & Dependencies                                     |
| 2    | Data and Model Preparation                               |
| 3    | Embedding Extraction with TTA                            |
| 4    | Build Validation Feature Database                        |
| 5    | Distorted Image Matching                                 |
| 6    | Thresholding and Result Storage                          |
| 7    | Evaluation and Analysis                                  |
| 8    | Visualization                                            |
| 9    | End                                                      |

**Flowchart:**

https://github.com/Nabanita315/Face-Intelligence-Tasks/blob/main/TaskB-Flowchart.png

**How to Run:**
1. Place your data in `Task_B/train/` and `Task_B/val/` with each person in a separate folder.  
   Place distorted images in a `distortion/` subfolder inside each person’s folder.
2. Open a terminal or Jupyter/VS Code and **change your working directory to `Task_B`**:
    ```bash
    cd Comys_Hackathon5/Task_B
    ```
3. Open and run `task-b.ipynb`.
4. The notebook expects the following structure:
    ```
    Task_B/
    ├── train/
    │   ├── person1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   │   └── distortion/
    │   │       ├── distorted1.jpg
    │   │       ├── distorted2.jpg
    │   │       └── ...
    │   ├── person2/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── ...
    │
    ├── val/
    │   ├── personx/
    │   │   ├── img_clean1.jpg
    │   │   ├── img_clean2.jpg
    │   │   ├── ...
    │   │   └── distortion/
    │   │       ├── distorted1.jpg
    │   │       ├── distorted2.jpg
    │   │       └── ...
    │   ├── persony/
    │   │   ├── img_clean1.jpg
    │   │   └── distortion/
    │   │       ├── distorted1.jpg
    │   │       └── ...
    │   └── ...
    │   └── task-b.ipynb
    ```
5. The default dataset paths (`./train`, `./val`) will work as long as you run the notebook from inside the `Task_B` folder.
6. Change the "**val_dir**" path if testing on another dataset.

---

## Evaluation Metrics

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-score

✅ Confusion Matrix (for Task A)

✅ Matching visualizations (for Task B)

## Requirements

    - Python 3.8+
    - torch
    - torchvision
    - facenet-pytorch
    - numpy
    - pillow
    - matplotlib
    - scikit-learn
    - tqdm
    - seaborn

Install requirements with:
```bash
pip install torch torchvision facenet-pytorch numpy pillow matplotlib scikit-learn tqdm seaborn
```

---

## Submission Guidelines
 
This repository includes:

✅ Clean, well-commented Jupyter notebooks for both tasks

✅ Training and validation results

✅ Easy-to-modify dataset paths

✅ Pretrained model weights (if applicable)

✅ A README.md with complete instructions

✅ Test-time script can be adapted to new data via test_dir or val_dir variables

## Notes

Always check your current working directory with:
    ```python
    import os
    print(os.getcwd())
    ```
...to ensure the notebook is running from the correct directory.

---
