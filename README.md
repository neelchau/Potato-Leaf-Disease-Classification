*Potato Leaf Disease Classification
---
**Project Overview
This project focuses on building a Convolutional Neural Network (CNN) model to accurately classify potato leaf diseases into three categories: 'Early Blight', 'Late Blight', and 'Healthy'. The goal is to provide a robust solution for early detection of these diseases, which can significantly aid in agricultural management and prevent crop loss.
---
**Dataset
The dataset used for this project was obtained from Kaggle (emmarex/plantdisease). It consists of images of potato leaves, categorized into:

Potato___Early_blight
Potato___Late_blight
Potato___healthy
The dataset was downloaded, unzipped, and preprocessed to prepare it for model training.
---
**Preprocessing and Data Augmentation
***Data Splitting
The dataset was split into training, validation, and testing sets with the following proportions:

Training Set: 80%
Validation Set: 10%
Test Set: 10%
Image Preprocessing
All images were resized to 256x256 pixels and rescaled to a range of [0, 1] to standardize the input for the neural network.

***Data Augmentation
To improve the model's generalization capabilities and reduce overfitting, various data augmentation techniques were applied to the training dataset, including:

Random horizontal and vertical flipping
Random rotations
Random zooming
Random brightness adjustments
Random translations
Random contrast adjustments
Gaussian noise addition
---
**Model Architecture
The CNN model is a sequential model built using TensorFlow/Keras. It comprises multiple convolutional layers followed by max-pooling layers to extract hierarchical features from the images. The architecture includes:

An initial Resizing and Rescaling layer (as part of the model for inference).
Multiple Conv2D layers (with 32 or 64 filters, 3x3 kernel, 'relu' activation) for feature extraction.
MaxPooling2D layers (2x2) for dimensionality reduction.
A Flatten layer to convert the 2D feature maps into a 1D vector.
A Dense hidden layer (with 64 units and 'relu' activation).
An output Dense layer (with 3 units and 'softmax' activation) for multi-class classification.
---
**Training
The model was compiled with:

Optimizer: Adam
Loss Function: SparseCategoricalCrossentropy
Metrics: accuracy
An EarlyStopping callback was used to monitor the val_loss with a patience of 10 epochs, ensuring the model stops training when performance on the validation set no longer improves, and restoring the best weights.

The training process ran for a maximum of 30 epochs, with a BATCH_SIZE of 32.
---
**Results
After training, the model achieved impressive performance:

Test Loss: Approximately 0.023
Test Accuracy: Approximately 99.06%
A confusion matrix was generated to visualize the classification performance across the three classes.
---
Deployment (Gradio Interface)
A simple web interface was built using Gradio to allow for easy testing and prediction. Users can upload an image of a potato leaf, and the interface will display the predicted class and confidence scores for each disease type
---
