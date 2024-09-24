markdown
# Face Age Prediction

This project implements a Convolutional Neural Network (CNN) to predict a person's age based on facial images. The model is trained on a dataset of face images and corresponding age labels. It uses the `TensorFlow` and `Keras` libraries for building and training the neural network.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [License](#license)

## Installation

Before running the project, make sure you have the following dependencies installed:

- Python 3.7+
- Numpy
- Pillow
- Scikit-learn
- TensorFlow (with Keras)
- Matplotlib (optional, for data visualization)
- Pandas
- Pickle

You can install the dependencies using:

```bash
pip install numpy pillow scikit-learn tensorflow matplotlib pandas
```

## Dataset

### Dataset Formation
The dataset consists of facial images categorized by age groups. For each image, the corresponding age is extracted from the folder name. Images are resized to 100x100 pixels and normalized by dividing by 255 for scaling between 0 and 1. The age labels are saved as a pickle file, while the images are saved as a NumPy array.

### Directory Structure
The facial images are stored in the following directory structure:

```
face_age/
    ├── 0/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── 1/
    │   ├── img3.jpg
    │   ├── img4.jpg
    └── ...
```

Where each folder name (e.g., `0`, `1`, ...) represents an age label.

### Dataset Extraction
- Images are loaded from the `.npy` file and labels are extracted from the `.pkl` file.
- The dataset is then shuffled randomly for better model generalization.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using Keras. The architecture consists of the following layers:

1. **Conv2D (64 filters, kernel size 3x3)**: ReLU activation, followed by a second Conv2D layer.
2. **MaxPooling2D**: Pooling layer to reduce the spatial dimensions.
3. **Conv2D (128 filters)**: ReLU activation, followed by another MaxPooling layer.
4. **Conv2D (256 filters)**: Final set of convolution layers with MaxPooling.
5. **Dropout**: Dropout of 40% to prevent overfitting.
6. **Flatten**: Flatten the output from the convolutional layers.
7. **Dense (128 units)**: Fully connected layer with ReLU activation.
8. **Dense (19 units)**: Final softmax output layer for age classification.

### Model Compilation

- Optimizer: `Adam`
- Loss Function: `Sparse Categorical Crossentropy`
- Metric: `Sparse Categorical Accuracy`

## Training the Model

To train the model, the data is split into a training set and a test set using an 80-20 split. The model is then trained over 20 epochs with a batch size of 150.

```python
model.fit(train_x, train_y, epochs=20, validation_data=(test_x, test_y), batch_size=150)
```

The model is saved after training:

```python
model.save('Models/Age2.h5')
```

## Making Predictions

To make predictions on new images:

1. Load the pre-trained model.
2. Preprocess the new image (resize and normalize).
3. Pass the image to the model for prediction.

Example:

```python
img = Image.open('Piks/suar_sir.jpg').convert('RGB')
img = img.resize((100, 100))
img = np.expand_dims(img, axis=0)
img = np.array(img) / 255

pred = model.predict(img)
predicted_age_group = np.argmax(pred, axis=1)
print(predicted_age_group)
```

## Results

The model predicts age groups as discrete labels. Each label corresponds to an age range (for example, label `0` could correspond to age 0-5, label `1` to 5-10, etc.). You can visualize predictions and test model performance using a validation dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
