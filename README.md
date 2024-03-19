# MNIST Classifier

This Python script, `MNIST_Classification.py`, implements a CNN classifier for the MNIST dataset, containing handwritten digits 0 to 9. It provides functions for loading data, building, training, and evaluating the model.

## Usage

1. Install Python 3.x, TensorFlow, and NumPy.
2. Run the script:

    ```bash
    python MNIST_Classification.py
    ```

## Functions

- **`get_dataset(training=True)`**: Loads MNIST dataset.
- **`print_stats(train_images, train_labels)`**: Prints dataset statistics.
- **`build_model()`**: Constructs a CNN model.
- **`train_model(model, train_images, train_labels, T)`**: Trains the model.
- **`evaluate_model(model, test_images, test_labels, show_loss=True)`**: Evaluates model accuracy.
- **`predict_label(model, test_images, index)`**: Predicts label for a specific image.

## Example

```python
from MNIST_Classification import *

train_images, train_labels = get_dataset(training=True)
print_stats(train_images, train_labels)
model = build_model()
train_model(model, train_images, train_labels, T=10)
test_images, test_labels = get_dataset(training=False)
evaluate_model(model, test_images, test_labels)
predict_label(model, test_images, index=0)
