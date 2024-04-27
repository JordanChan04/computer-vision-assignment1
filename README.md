# Neural Network Classifier for Fashion-MNIST

This repository contains code for training and testing a three-layer neural network classifier from scratch using numpy. The model is trained on the Fashion-MNIST dataset to perform image classification.

## Dependencies
- Python (>=3.6)
- NumPy
- idx2numpy (for loading the dataset)
- matplotlib (for visualization)

Install the dependencies using pip:
```bash
pip install numpy idx2numpy matplotlib
```

## Data Preparation
1. Download the Fashion-MNIST dataset from [here](https://github.com/zalandoresearch/fashion-mnist).
2. Extract the downloaded files to a directory.
3. Ensure the following files are present in the directory:
   - `train-images-idx3-ubyte`: Training images
   - `train-labels-idx1-ubyte`: Training labels
   - `t10k-images-idx3-ubyte`: Test images
   - `t10k-labels-idx1-ubyte`: Test labels

## Training Procedure
1. Open the terminal and navigate to the directory containing the downloaded dataset and the code files.
2. Execute the training script with the desired hyperparameters:
   ```bash
   python train.py --lr <learning_rate> --hidden_size <hidden_layer_size> --l2_penalty <l2_penalty> --epochs <num_epochs>
   ```
   Replace `<learning_rate>`, `<hidden_layer_size>`, `<l2_penalty>`, and `<num_epochs>` with appropriate values.

## Testing Procedure
1. Ensure that the model has been trained using the training procedure mentioned above.
2. Execute the testing script to evaluate the trained model on the test set:
   ```bash
   python test.py --model_path <path_to_saved_model>
   ```
   Replace `<path_to_saved_model>` with the path to the saved model weights.

## Example Usage
### Training
```bash
python train.py --lr 0.1 --hidden_size 30 --l2_penalty 0.001 --epochs 1000
```

### Testing
```bash
python test.py --model_path final_model_params.npz
```
