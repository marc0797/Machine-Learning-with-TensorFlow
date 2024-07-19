## Multi-class classification with neural networks

# In this exercise, we explore a multi-class classification
# problem through the MNIST dataset using neural networks.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

np.set_printoptions(linewidth=200)

def build_model(learning_rate, 
        n_hidden_layers=1, n_neurons=(32,), dropout_rate=0.0):
    """ Create and compile a deep neural network. """

    model = tf.keras.models.Sequential()

    # Flatten the 28x28 images to a 784x1 vector
    model.add(layers.Flatten(input_shape=(28, 28)))

    # First hidden layer with ReLU activation
    for i in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons[i], activation='relu'))
        model.add(layers.Dropout(dropout_rate))

    # Output layer with 10 units and softmax activation
    # IMPORTANT: With softmax, the number of units must 
    # match the number of classes. 
    # In this case 0-9 -> 10 classes.
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model, we are only interested in accuracy
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train, epochs, bs, val_split):
    """ Train the model with the data. """

    history = model.fit(x=x_train, y=y_train, batch_size=bs, 
                        epochs=epochs, shuffle=True,
                        validation_split=val_split)
    
    # Gather the history of each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
    """ Plot a curve of one or more classification metrics. """
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for metric in list_of_metrics:
        x = hist[metric]
        y = hist['val_' + metric]
        plt.plot(epochs[1:], x[1:], label=metric)
        plt.plot(epochs[1:], y[1:], label='val_' + metric)

    plt.legend()
    plt.show()

def main(lr, n_epochs, bs, val_split, nh_layers=1, nh_neurons=(32,), drop_rate=0.0):
    # Load the MNIST dataset from keras or the data folder
    # Uncomment one of the two options below to load the data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # train_data = pd.read_csv('data/mnist_train_small.csv')
    # x_train = train_data.iloc[:, 1:].values
    # y_train = train_data.iloc[:, 0].values
    # test_data = pd.read_csv('data/mnist_test.csv')
    # x_test = test_data.iloc[:, 1:].values
    # y_test = test_data.iloc[:, 0].values

    # # Reshape into 28x28 images
    # x_train = x_train.reshape(-1, 28, 28)
    # x_test = x_test.reshape(-1, 28, 28)

    # Show the dataset
    plt.imshow(x_train[4632])
    plt.show()

    # Normalize the data (from 0-255 to 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build the model
    model = build_model(lr)

    # Train the model
    epochs, hist = train_model(model, x_train, y_train, n_epochs, bs, val_split)

    # Plot the accuracy
    list_of_metrics = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics)

    # Evaluate the model
    model.evaluate(x_test, y_test, batch_size=bs)

if __name__ == "__main__":
    # Set the hyperparameters
    lr = 0.003
    epochs = 100
    bs = 4000
    val_split = 0.2

    # Additional hyperparameters
    nh_layers = 2
    nh_neurons = (256,128,)
    drop_rate = 0.2

    main(lr, epochs, bs, val_split, nh_layers, nh_neurons, drop_rate)