# Neural Network Classifier for Fashion-MNIST

This repository contains code for training and testing a three-layer neural network classifier from scratch using numpy. The model is trained on the Fashion-MNIST dataset to perform image classification.

## Data Preparation

To train and test the model, you need to download the MNIST dataset. The `load_mnist_data()` function provided in the code can be used to load the data. Ensure that you have the following files in the project directory:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

These files contain the images and labels for both the training and test sets.

## Training Procedure

To train the model, follow these steps:

1. Import the necessary functions and modules from the provided code.
2. Load the MNIST data using the `load_mnist_data()` function.
3. Split the training dataset into training and validation sets using the `custom_train_validation_split()` function.
4. Choose hyperparameters for training, such as learning rate, hidden layer size, and L2 penalty.
5. Use the `train_with_metrics()` function to train the model with the selected hyperparameters.
6. Optionally, tune hyperparameters using the `parameter_tuning()` function.

## Testing Procedure

To test the trained model, follow these steps:

1. Load the test dataset using the `load_mnist_data()` function.
2. Import the necessary functions and modules from the provided code.
3. Load the trained model parameters from the saved file using `np.load()`.
4. Use the `forward()` function to perform forward propagation on the test dataset.
5. Calculate accuracy and other metrics using the `accuracy()` function.

## Example Usage

Here's an example of how to train and test the model:

```python
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from model import *

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

# Split the original training dataset into training and validation sets
train_images, train_labels, val_images, val_labels = custom_train_validation_split(train_images, train_labels, validation_ratio=0.2)

# Define hyperparameters
lr = 0.1
hidden_size = 20
l2_penalty = 0.1
epochs = 1000
batch_size = 256

# Train the model
train_losses, train_accuracies, test_losses, test_accuracies = train_with_metrics(
    (train_images, train_labels), train_labels, (test_images, test_labels), test_labels,
    epochs=epochs, batch_size=batch_size, lr=lr, l2_penalty=l2_penalty, save_path="final_model_params.npz"
)

# Test the model
saved_params = np.load("final_model_params.npz")
final_params = {
    'w1': saved_params['w1'],
    'b1': saved_params['b1'],
    'w2': saved_params['w2'],
    'b2': saved_params['b2'],
    'w3': saved_params['w3'],
    'b3': saved_params['b3']
}
res_test, _, _ = forward(test_images, final_params)
test_accuracy = accuracy(res_test, test_labels)
print("Test Accuracy:", test_accuracy)
```

## Visualizations

You can visualize the training process and learned parameters using the provided functions in the code. For example:

- Plot training and test loss curves.
- Plot accuracy curves.
- Visualize learned parameters such as weights and biases.

Refer to the provided code for more details on how to generate visualizations.
