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
The goal is to teach a model to distinguish between male and female faces in pictures.

**Highlights:**  
- employs a ResNet18 model that has been pretrained.
- incorporates data augmentation.
- uses weighted sampling to address class imbalance.
- evaluates performance using the **confusion matrix, F1-score, recall, accuracy, and precision**.

**Workflow:**

[TaskA-workflow.md](https://github.com/Nabanita315/Face-Intelligence-Tasks/blob/main/TaskA-workflow.md)

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


**Key points:**

✅ Evaluation Metrics 

**1. Accuracy:**
- Training and validation accuracy are printed after every epoch.
- Calculated as:
correct_predictions / total_samples

**2. Classification Report (sklearn.metrics.classification_report):**
   Precision, Recall, F1-score, and Support for each class are printed for both:
   - Validation set (after training)
   - Test set (in test_model())

**3. Confusion Matrix (ConfusionMatrixDisplay):**
   Visualizes the true vs. predicted classes for:
   -Validation set
   -Test set

✅ Model Weights Used

**1.Base Model:**
torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
- This loads ImageNet-pretrained weights for the ResNet18 model.

**2. Modified Head:**
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, num_classes)
)
- The final layer is replaced to fit the binary classification task (num_classes = 2).

**3. Saved Best Weights:**
- During training, the model with the best validation accuracy is saved:
torch.save(model.state_dict(), "best_gender_model.pth")

**4. Testing:**
- The model is reloaded using the best saved weights:
model.load_state_dict(torch.load("best_gender_model.pth", map_location=device))
---

## Task B: Face Recognition with Distorted Images

**Objective:**  
Match face images (including distorted versions) to their corresponding identities.

**Highlights:**
- For embedding extraction, a pretrained **FaceNet** model is used.
- Uses **cosine similarity** to match distorted images to clean ones.
- **Top-1 Accuracy, Macro-averaged F1 score** is used for evaluation.
- **Visualizations** of the corresponding results are provided.

**Workflow:**

https://github.com/Nabanita315/Face-Intelligence-Tasks/blob/main/TaskB-overflow.md

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

**Key points:**

✅ Evaluation Metrics 

**1. Top-1 Accuracy:**
top1_acc = accuracy_score(true_labels, pred_labels)
- Measures whether the predicted identity matches the true identity.
- Match is only accepted if:
    - Predicted name matches true name
    - Cosine similarity > threshold (default = 0.65)

**2. Macro F1 Score:**
macro_f1 = f1_score(true_valid, pred_valid, average='macro')
   - Macro-averaged across all valid (accepted) predictions
   - Useful in imbalanced class settings
   - Ignores mismatches and thresholded-out examples

**3. Cosine Similarity:**
  cosine_similarity([embedding], embeddings)
   - Used to match a distorted face to gallery embeddings.
   - Threshold (default 0.65) determines if a match is valid
   - Plotted with histograms.
     
**4. Threshold Tuning Curve:**
   threshold_curve(results)
   - This helps determine an optimal cosine similarity threshold.

**5.  Histogram of Similarity Scores**
  plt.hist(match_scores, ...)
  plt.hist(mismatch_scores, ...)
   - Distribution of matches vs. mismatches with a threshold reference line.

✅ Model Weights Used

**Pretrained Model:**
- facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
- This loads pretrained FaceNet weights trained on the VGGFace2 dataset. The weights come from the FaceNet model trained to embed faces where similar faces are closer.

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
## Submission Guidelines
 
This repository includes:

✅ Clean, well-commented Jupyter notebooks for both tasks

✅ Training and validation results

✅ Easy-to-modify dataset paths

✅ Pretrained model weights (if applicable)

✅ A README.md with complete instructions

✅ Test-time script can be adapted to new data via test_dir or val_dir variables

## Notes

To make sure the notebook is operating from the correct directory, always use the following command to verify your current working directory:

    ```python
    import os
    print(os.getcwd())
    ```
    
---
