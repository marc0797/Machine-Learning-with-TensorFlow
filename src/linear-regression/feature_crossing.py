## Linear regression on real dataset with feature crossing

# In this excercise we will be using feature crossing 
# to account for non-linearities in the data.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from matplotlib import pyplot as plt

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

# Define a function that builds an empty model.
def build_model(inputs, outputs, learning_rate):
    """ Create and compile the model. """    

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model and configure training to minimize MSE.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

# Function which trains the model from the examples you pass.
def train_model(model, data, epochs, batch_size, label):
    """ Train the model with the data. """

    # Set up a dictionary to hold the data.
    features = {name:np.array(value) for name, value in data.items()}
    labels = np.array(features.pop(label))

    # Feed the feature values and the label values to the model.
    history = model.fit(x=features,
                        y=labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True)

    # List of epochs
    epochs = history.epoch

    # Gather the history of each epoch.
    hist = pd.DataFrame(history.history)

    # Gather root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse

def plot_loss(epochs, training_error):
    """ Plot the loss curve, which shows loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, training_error, label="Loss")
    plt.legend()
    plt.ylim([training_error.min() * 0.97, training_error.max() * 1.03])
    plt.show()

# Function to reset the model, train it and evaluate it
def reset(training_data, test_data, inputs, outputs, lr, n_epochs, bs, label):
    # Build the model.
    model = None
    model = build_model(inputs, outputs, lr)

    # Train the model
    epochs, rmse = train_model(model, training_data, n_epochs, bs, label)

    # Print a summary of the model
    model.summary(expand_nested=True)

    # Plot the loss curve
    plot_loss(epochs, rmse)

    # Evaluate the model
    print("\nEvaluating the model on the test set:")
    print("--------------------------------------\n")
    x_test = {name:np.array(value) for name, value in test_data.items()}
    y_test = np.array(x_test.pop(label))
    model.evaluate(x=x_test, y=y_test, batch_size=bs)

    return model

def main(n_epochs=10, lr=0.01, bs=10):
    # Load the dataset
    training_data_path = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
    training_data = pd.read_csv(training_data_path)
    test_data_path = "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
    test_data = pd.read_csv(test_data_path)

    scaling_factor = 1000.0

    # Scale the training and test data.
    training_data["median_house_value"] /= scaling_factor
    test_data["median_house_value"] /= scaling_factor

    # Shuffle the examples
    training_data = training_data.reindex(np.random.permutation(training_data.index))

    label = "median_house_value"

    # Keras Input features
    inputs = {
        'latitude': tf.keras.Input(shape=(1,), name='latitude',
            dtype=tf.float32),
        'longitude': tf.keras.Input(shape=(1,), name='longitude',
            dtype=tf.float32)
    }

    # Floating point representation
    print("\nFloating point representation of latitude and longitude:")
    print("--------------------------------------------------------\n")

    # Create a concatenated layer of the two inputs
    concatenate_layer = tf.keras.layers.Concatenate()([inputs['latitude'], inputs['longitude']])

    # Add a dense layer
    dense_layer = layers.Dense(units=1, name='dense_layer')(concatenate_layer)

    outputs = {
        'dense_layer': dense_layer
    }

    model = reset(training_data, test_data, inputs, outputs, lr, n_epochs, bs, label)

    # Bucket representation of latitude and longitude
    print("\nBin representation of latitude and longitude:")
    print("--------------------------------------------\n")
    # Resolution of the latitude and longitude bins
    resolution = 1.0
    latitude_boundaries = list(np.arange(int(min(training_data["latitude"])),
                                        int(max(training_data["latitude"])), 
                                        resolution))
    longitude_boundaries = list(np.arange(int(min(training_data["longitude"])),
                                         int(max(training_data["longitude"])), 
                                         resolution))
    
    # First, we discretize the latitude and longitude in bins
    discretized_latitude = tf.keras.layers.Discretization(
        bin_boundaries=latitude_boundaries,
        name='discretization_latitude')(inputs.get('latitude'))
    
    discretized_longitude = tf.keras.layers.Discretization(
        bin_boundaries=longitude_boundaries,
        name='discretization_longitude')(inputs.get('longitude'))
    
    # Then, we encode the bins using one-hot encoding. The
    # number of categories is len(boundaries) plus one.
    encoded_latitude = tf.keras.layers.CategoryEncoding(
        num_tokens=len(latitude_boundaries) + 1,
        output_mode='one_hot',
        name='category_encoding_latitude')(discretized_latitude)
    encoded_longitude = tf.keras.layers.CategoryEncoding(
        num_tokens=len(longitude_boundaries) + 1,
        output_mode='one_hot',
        name='category_encoding_longitude')(discretized_longitude)

    # Concatenate both layers into a single tensor
    concatenate_layer = tf.keras.layers.Concatenate()(
        [encoded_latitude, encoded_longitude])

    dense_layer = layers.Dense(units=1, name='dense_layer')(concatenate_layer)

    outputs = {
        'dense_layer': dense_layer
    }

    model = reset(training_data, test_data, inputs, outputs, lr, n_epochs, bs, label)

    # Feature cross representation
    print("\nFeature cross representation of latitude and longitude:")
    print("-------------------------------------------------------\n")
    # Create a feature cross of latitude and longitude, by multiplying
    # the two discretized layers, obtaining a 2D grid.
    feature_cross = tf.keras.layers.HashedCrossing(
        num_bins=len(latitude_boundaries) * len(longitude_boundaries),
        output_mode="one_hot",
        name="crossed_lat_lon")([discretized_latitude, discretized_longitude])

    dense_layer = layers.Dense(units=1, name='dense_layer')(feature_cross)

    outputs = {
        'dense_layer': dense_layer
    }

    model = reset(training_data, test_data, inputs, outputs, lr, n_epochs, bs, label)

    # Decrease the resolution of the bins
    print("\nDecreased resolution of the bins:")
    print("---------------------------------\n")
    resolution = 0.4

    latitude_boundaries = list(np.arange(int(min(training_data["latitude"])),
                                        int(max(training_data["latitude"])), 
                                        resolution))
    longitude_boundaries = list(np.arange(int(min(training_data["longitude"])),
                                         int(max(training_data["longitude"])), 
                                         resolution))

    discretized_latitude = tf.keras.layers.Discretization(
        bin_boundaries=latitude_boundaries,
        name='discretization_latitude')(inputs.get('latitude'))
    discretized_longitude = tf.keras.layers.Discretization(
        bin_boundaries=longitude_boundaries,
        name='discretization_longitude')(inputs.get('longitude'))
    
    feature_cross = tf.keras.layers.HashedCrossing(
        num_bins=len(latitude_boundaries) * len(longitude_boundaries),
        output_mode="one_hot",
        name="crossed_lat_lon")([discretized_latitude, discretized_longitude])

    dense_layer = layers.Dense(units=1, name='dense_layer')(feature_cross)

    outputs = {
        'dense_layer': dense_layer
    }

    model = reset(training_data, test_data, inputs, outputs, lr, n_epochs, bs, label)
    

if __name__ == "__main__":
    # Play with hyperparameters to adjust the model.
    epochs = 35
    learning_rate = 0.04
    batch_size = 100

    main(n_epochs=epochs, lr=learning_rate, bs=batch_size)