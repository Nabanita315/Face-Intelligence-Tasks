# Comys Hackathon 5: Face-Intelligence-Tasks

This repository contains solutions for two computer vision tasks:

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
```

---

## Task A: Gender Classification

**Objective:**  
Train a model to classify face images as either male or female.

- Uses a pretrained ResNet18 model with data augmentation and weighted sampling for class imbalance.
- Evaluates performance with classification report and confusion matrix.

**How to Run:**
1. Place your data in `Task_A/train/` and `Task_A/val/` with subfolders `male/` and `female/`.
2. Open a terminal or Jupyter/VS Code and **change your working directory to `Task_A`**:
    ```bash
    cd Comys_Hackathon5/Task_A
    ```
3. Open and run `task-a.ipynb`.
4. The notebook expects the following structure:
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
7. For evaluating on a new dataset, change the path of "test_dir" with your data directory path.

---

## Task B: Face Recognition with Distorted Images

**Objective:**  
Match a face image (including distorted versions) to the correct person.

- Uses a pretrained FaceNet model for embedding extraction.
- Matches distorted images to clean ones using cosine similarity.
- Evaluates with accuracy, F1 score, and visualizations.

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
6. Change the "val_dir" path with your test dataset path if needed.

---

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

## Notes

- Update the dataset paths in the notebooks only if your folder structure is different.
- Both notebooks are well-commented for easy understanding and modification.
- Always check your current working directory with:
    ```python
    import os
    print(os.getcwd())
    ```
  and make sure it matches the folder containing your notebook and data.

---
