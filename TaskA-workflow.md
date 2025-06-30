# Gender Classification Workflow (ResNet18 Based)

This document outlines the step-by-step workflow for gender classification using a ResNet18 model, as implemented in `task-a.ipynb`.

---

## 1. Setup & Imports

- Import all necessary libraries:
  - PyTorch, torchvision, numpy, scikit-learn, matplotlib, etc.
- Set parameters:
  - Number of classes: **2** (male/female)
  - Batch size, number of epochs, learning rate
- Check if GPU is available and select the device (GPU or CPU).

---

## 2. Image Transformations

- **Training images**:
  - Resize to 224x224
  - Apply random horizontal flips and small rotations
  - Adjust brightness and contrast
  - Convert to tensor and normalize
- **Validation/Test images**:
  - Only resize, convert to tensor, and normalize

---

## 3. Model Definition

- Use a pre-trained **ResNet18** model.
- Replace the final layer for gender classification (2 output units).
- Add dropout for better generalization.

---

## 4. Training Preparation

- Load train and validation datasets using folder structure.
- Handle class imbalance:
  - Compute class weights
  - Use a weighted random sampler for balanced mini-batches
- Create data loaders for efficient data feeding.

---

## 5. Model Training

- For each epoch:
  - Train the model on the training set (with augmentation and weighted sampling)
  - After each epoch, evaluate accuracy on the validation set
  - Save the model weights if the validation accuracy improves

---

## 6. Evaluation & Saving

- After training:
  - Save the best performing model
  - Evaluate the model on the validation set:
    - Print a classification report (precision, recall, f1-score)
    - Plot and display a confusion matrix

---

## 7. Testing

- Load the saved best model.
- Evaluate on the test set (can be the same as validation if no separate test set).
- Print classification report and plot confusion matrix for test results.

---

## 8. How To Use

- Set the correct folder paths for your `train`, `val`, and `test` data.
- Run the notebook.
- The notebook will train, validate, test, and display all evaluation results.

---

**Summary:**  
You provide gender-labeled face images; the notebook takes care of robust model training (with class imbalance handling) and provides detailed evaluation metrics — just by specifying your data folders.
