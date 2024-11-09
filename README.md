Self-Driving Car Project

This project aims to implement a basic self-driving car model using deep learning and computer vision. It processes road images to predict steering angles, allowing a car to drive autonomously. The model is trained and tested using simulated driving data, and it uses real-time telemetry data for testing on a server.

Project Structure:

train.py - Contains code for data preprocessing, model training, and saving the trained model.
test.py - Used for real-time testing and prediction, receiving telemetry data and using the model to predict steering angles.
utils.py - Contains helper functions for data augmentation, preprocessing, and batch generation, as well as the model architecture.

Steps to Run the Project:

1. Data Preparation:

Data Import: Place your driving log CSV and image data in the specified directory (e.g., myData).
Balance Data: balanceData() function reduces data imbalance by limiting the number of samples per steering angle bin.
Load Data: Use the loadData() function to extract image paths and steering values.

2. Model Training:

Run train.py to start the training process.
Training and validation sets are created with an 80/20 split.
Model is trained using a custom generator (batchGen) for data augmentation and preprocessing.
An early stopping callback is used to prevent overfitting.
Save the Model: The trained model is saved as model1.h5.

3. Model Testing:

Run test.py to start the Flask server for testing.
Receives telemetry data through SocketIO.
Uses the preProcess function to preprocess images in real time.
Predicts steering angles and transmits commands to control the car's movement.
Run Simulation: This code is designed for testing within a driving simulator that communicates via SocketIO.
Installation and Dependencies
Ensure that all dependencies are installed:

Installation and Dependencies:

pip install tensorflow keras numpy pandas scikit-learn imgaug flask socketio eventlet imbalanced-learn opencv-python-headless matplotlib

Project Components:

Model Architecture:

The model is based on a convolutional neural network inspired by Nvidia’s architecture for self-driving cars. It includes:

Convolutional layers with ELU activation and Batch Normalization.
Dense layers with Dropout for regularization.
Final layer outputs a single value representing the steering angle.

Data Augmentation:

To improve model generalization, several augmentation techniques are applied, such as:

Pan: Shifts the image randomly.
Zoom: Randomly zooms into the image.
Brightness: Randomly changes image brightness.
Flip: Horizontally flips the image with a 50% probability.
Real-Time Control
The test.py file sets up a Flask server that receives image data, processes it, and predicts steering angles in real time.
