# Real-time Mask Detection using CNN and OpenCV

**Done by:** Aasem Nasir Al-Dgree  
**ID:** 202170208

## Overview
This project implements a real-time face mask detection system using a Convolutional Neural Network (CNN) and OpenCV. The model is trained to classify whether a person is wearing a mask or not. The trained model is integrated with a real-time webcam feed to detect faces and classify them as "Mask" or "No Mask".

## Project Structure
```plaintext
├── data/
│   ├── with_mask/
│   ├── without_mask/
├── mask_detection_model/
├── main.py
├── Training model.ipynb
├── Report.word
├── README.md
└── requirements.txt
data/: Contains images for training (with mask and without mask).
mask_detection_model/: Contains the trained model file.
main.py: Main script for mask detection using the webcam.
README.md: Documentation for the project.
requirements.txt: List of required Python libraries.
```

## Requirements
To install the required libraries, run:

```plaintext
pip install -r requirements.txt
```
The main libraries used include:

TensorFlow/Keras (for building and training the CNN model)
OpenCV (for real-time video capture and face detection)
NumPy (for handling arrays)
Matplotlib (for visualizing data)

# How to Run
## Training the Model:
The model is trained on a dataset of images with and without masks. You can modify the training script in the project and retrain the model with your own dataset.

## Real-time Detection:
Run the main.py script to start the webcam-based mask detection.
The webcam feed will show bounding boxes around detected faces along with labels indicating whether the person is wearing a mask or not.

To start the real-time mask detection, run:

```plaintext
python main.py
```
## Stopping the Detection:
Press q to stop the webcam feed and exit the program.

## Model Training
The model is a simple CNN architecture consisting of:

Convolutional Layers
MaxPooling Layers
Fully Connected Dense Layers
Dropout layers for preventing overfitting
The model is compiled using the Adam optimizer and the sparse_categorical_crossentropy loss function. The dataset is divided into training and testing sets with an 80/20 split. After 5 epochs, the model achieves good accuracy in mask detection.

# Future Enhancements
Improve the accuracy of the mask detection by using more advanced deep learning models such as ResNet or EfficientNet.
Add a notification system to alert if a person without a mask is detected.
Implement multi-face detection for crowded environments.
