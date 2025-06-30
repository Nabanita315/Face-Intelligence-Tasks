1. **Set Hyperparameters & Device**
   - Define the number of classes (`num_classes=2`), batch size, number of epochs, and learning rate.
   - Set the device to GPU if available; otherwise, use CPU.

2. **Define Image Transformations**
   - **Training Data:** Resize images, apply augmentation (random flip, rotation, color jitter), convert to tensor, and normalize.
   - **Validation/Test Data:** Resize images, convert to tensor, and normalize (without augmentation).

3. **Model Preparation**
   - Load a pretrained ResNet18 model.
   - Replace the final layer with a dropout and a new linear layer for two-class output.
   - Move the model to the selected device.

4. **Data Loading**
   - Load training and validation datasets from their directories using `ImageFolder` and the defined transforms.

5. **Handle Class Imbalance**
   - Calculate class frequencies in the training data.
   - Compute sample weights inversely proportional to class frequencies.
   - Use `WeightedRandomSampler` for balanced training batches.

6. **Create DataLoaders**
   - Use the weighted sampler in the training DataLoader.
   - Use batch size 1 and no shuffle for the validation DataLoader.

7. **Training Loop**
   - For each epoch:
     - Train on all training batches: perform forward pass, compute loss, backward pass, and optimizer step.
     - Calculate training accuracy.
     - Evaluate the model on the validation set and calculate validation accuracy.
     - Save the model weights if validation accuracy improves.

8. **Evaluate on Validation Set**
   - After training, reload the best model weights.
   - Predict on the entire validation set.
   - Print a classification report and display the confusion matrix.

9. **Testing**
   - Load the test dataset.
   - Load the saved best model.
   - Predict on the test set.
   - Print a classification report and display the confusion matrix.

10. **Main Execution**
    - Set the paths for train, validation, and test directories.
    - Call the training function, then the testing function.
11. **End**
