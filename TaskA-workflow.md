1. Get Ready

2. Decide how many times to train the model, how big each batch should be, and whether to use your computer’s GPU or just the CPU.
Prep the Photos

3. For training, mix things up a bit! Randomly flip, rotate, or brighten the images so the model learns better.
For checking the model (validation/testing), just resize and keep them normal—no funny business.
Load the Model

Grab a ResNet18 model, which already knows a lot about images.
Swap out its last part, so it can decide between just “male” and “female.”
Read the Data

Load photos from your folders: one for training, one for checking (validation), and maybe one for final testing.
Fix Any Imbalances

Sometimes you have more photos of one gender than the other. To avoid bias, give more “weight” to the underrepresented group so the model pays extra attention to them.
Train the Model

Feed the model batches of photos. It makes guesses and learns from mistakes, over and over.
After each round, check how well it’s doing on the validation photos.
If it gets better, save the model’s “brain” (weights).
Check Results

When training is done, see how well the best model does on the validation set.
Print out a report: How often was it right? Where did it get confused?
Show a confusion matrix: a visual way to see where the model made mistakes.
Final Test

Use the saved best model to predict on the test set (new, unseen photos).
Print another report and confusion matrix for these test results.
Done!

Now you know how well your model can tell genders apart from photos.

---

In short:
You prepare your image data, balance the classes, train a smart model with some extra “practice” (augmentation), keep the best version, and finally see how it does on new photos. The process is careful to avoid bias and checks its own performance step by step!

