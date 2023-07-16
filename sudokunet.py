import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import tensorflow as tf # Machine learning library
from tensorflow import keras # Library for neural networks
from tensorflow.keras.datasets import mnist # MNIST data set
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from time import time
import matplotlib.pyplot as plt
 
# the Minst dataset is a dataset of number 0 to 9 to train a model that can recognize digits
num_classes = 10 
 
def generate_neural_network(x_train):
  model = keras.Sequential()

  # We decided to use 3 convolutional layers since this gave the model a reasonable amount
  # of insight on the digits without overfitting
  # We used batch normalization to speed up the training process
  # We used dropout in order to allow our model to train on varied architectures

  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(units=256, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(units=num_classes, activation='softmax'))

  return model

# KERAS requires data to be in 4 dimensions
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, np.append(x_train.shape, (1)))
x_test = np.reshape(x_test, np.append(x_test.shape, (1)))

# Normalize image intensity values to a range between 0 and 1 (from 0-255)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

# Perform one-hot encoding since there is no distance between classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the neural network
model = generate_neural_network(x_train)

# Configure the neural network
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    
start = time()
history = model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test), shuffle=True)
training_time = time() - start
print(f'Training time: {training_time}')
    
# A measure of how well the neural network learned the training data
# The lower, the better
print("Minimum Loss: ", min(history.history['loss']))

# A measure of how well the neural network did on the validation data set
# The lower, the better
print("Minimum Validation Loss: ", min(history.history['val_loss']))

# Maximum percentage of correct predictions on the training data
# The higher, the better
print("Maximum Accuracy: ", max(history.history['accuracy']))

# Maximum percentage of correct predictions on the validation data
# The higher, the better
print("Maximum Validation Accuracy: ", max(history.history['val_accuracy']))
    
# Plot the key statistics
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Mean Squared Error for the Neural Network on the MNIST Data")  
plt.ylabel("Mean Squared Error")
plt.xlabel("Epoch")
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 
'Validation Accuracy'], loc='center right') 
plt.show() # Press Q to close the graph
    
# Save the neural network in Hierarchical Data Format version 5 (HDF5) format
model.save('mnist_nnet.h5')