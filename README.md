# Brain Tumor Detection Using Deep Learning

This project aims to detect brain tumors from MRI images using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The model is trained to classify MRI images as either containing a tumor or not. This repository contains the code for training the model, testing it on new images, and deploying it as a web application using Flask.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Web Application](#web-application)
- [Usage](#usage)
- [Results](#results)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sarthakraut16/Brain-Tumor-Detection-using-deep-learning.git
   cd Brain-Tumor-Detection-using-deep-learning
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset should be structured as follows:
```
datasets/
    no/
        image1.jpg
        image2.jpg
        ...
    yes/
        image1.jpg
        image2.jpg
        ...
```
- `no/` directory contains images without brain tumors.
- `yes/` directory contains images with brain tumors.

## Model Training

The model is trained using the `mainTrain.py` script.

```python
python mainTrain.py
```

This script will:
1. Load and preprocess the dataset.
2. Split the dataset into training and testing sets.
3. Normalize the image data.
4. Build and compile a CNN model.
5. Train the model on the training data.
6. Save the trained model as `BrainTumor10EpochsCategorical.h5`.
7. Plot and save the training history.

## Model Testing

To test the model on a new image, use the `mainTest.py` script.

```python
python mainTest.py
```

This script will:
1. Load the saved model.
2. Read the input image.
3. Preprocess the image.
4. Predict if the image contains a tumor.
5. Print the prediction result.

## Web Application

The web application is built using Flask. It allows users to upload an MRI image and get a prediction along with confidence, precautions, and home treatment suggestions.

### Running the Web Application

1. Ensure the model is saved as `BrainTumorDetectionModel.h5`.
2. Start the Flask app by running:

```python
python app.py
```

3. Open a web browser and navigate to `http://127.0.0.1:5000/`.

### Web Interface

The web interface allows users to:
- Upload an MRI image.
- View the uploaded image.
- Get a prediction result along with confidence.
- View precautions and home treatment suggestions if a tumor is detected.

## Usage

1. Navigate to the home page.
2. Upload an MRI image.
3. Click on the "Predict!" button.
4. View the results, including the prediction, accuracy, precautions, and home treatment suggestions.

## Results

The model's performance is tracked during training, and accuracy/loss plots are generated and saved as `training_history.png`.

##Thank You
