## Neural Networks: Introduction

# Using the california housing dataset, we will build
# a neural network to predict the median house value.

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

# Adjust the granularity of reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def build_model(inputs, outputs, lr):
    """ Create and compile a simple linear regression model. """    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model and configure training to minimize MSE.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    return model

def train_model(model, data, epochs, bs, label, val_split):
    """ Train the model with the data. """

    # Split the data into features and labels
    features = {name:np.array(value) for name, value in data.items()}
    label = np.array(features.pop(label))

    # Normalize the labels
    median_house_value = tf.keras.layers.Normalization(axis=None)
    median_house_value.adapt(np.array(data['median_house_value']))

    label = median_house_value(label)

    # Train the model
    history = model.fit(x=features, y=label, batch_size=bs, 
                        epochs=epochs, shuffle=True,
                        validation_split=val_split)
    
    # Gather the history of each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist['mean_squared_error']

    return epochs, mse, history.history

def plot_loss_curve(epochs, error_training, error_validation):
    """ Plot the loss curve, which shows loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    # Skip the first epoch to give the graph a cleaner look.
    plt.plot(epochs[1:], error_training[1:], label="Training Loss")
    plt.plot(epochs[1:], error_validation[1:], label="Validation Loss")
    plt.legend()

    merged_error = error_training.tolist()[1:] + error_validation[1:]
    y_max = max(merged_error)
    y_min = min(merged_error)
    y_diff = y_max - y_min

    plt.ylim(y_min - y_diff*0.05, y_max + y_diff*0.05)
    plt.show()

def linear_regression_output(layers):
    """ Create a linear regression output layer. """
    
    dense_output = tf.keras.layers.Dense(units=1, 
                                name='dense_output')(layers)

    outputs = {
        'dense_output': dense_output
    }

    return outputs

def dnn_output(layers, units=(20, 12)):
    """ Create a DNN output layer. """

    # Hidden layer 1
    dense_output = tf.keras.layers.Dense(
                    units=units[0],
                    activation='relu',
                    name='hidden_layer_1')(layers)

    # Hidden layer 2
    dense_output = tf.keras.layers.Dense(
                    units=units[1],
                    activation='relu',
                    name='hidden_layer_2')(dense_output)
    
    # Output layer
    dense_output = tf.keras.layers.Dense(
                    units=1,
                    name='dense_output')(dense_output)

    outputs = {
        'dense_output': dense_output
    }

    return outputs

def main(learning_rate=0.01, n_epochs=15, batch_size=1000, validation_split=0.2):
    # Load the dataset
    training_data = pd.read_csv('data/california_housing_train.csv')
    testing_data = pd.read_csv('data/california_housing_test.csv')

    # Shuffle the data
    training_data = training_data.reindex(np.random.permutation(training_data.index))

    # Input layer definition
    inputs = {
        'latitude': tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                            name='latitude'),
        'longitude': tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                            name='longitude'),
        'median_income': tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                            name='median_income'),
        'population': tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                            name='population')
    }

    # Normalize data as they are on different scales
    latitude = tf.keras.layers.Normalization(
        name='normalized_latitude', axis=None)
    latitude.adapt(training_data['latitude'])
    latitude = latitude(inputs.get('latitude'))
    
    longitude = tf.keras.layers.Normalization(
        name='normalized_longitude', axis=None)
    longitude.adapt(training_data['longitude'])
    longitude = longitude(inputs.get('longitude'))

    median_income = tf.keras.layers.Normalization(
        name='normalized_median_income', axis=None)
    median_income.adapt(training_data['median_income'])
    median_income = median_income(inputs.get('median_income'))

    population = tf.keras.layers.Normalization(
        name='normalized_population', axis=None)
    population.adapt(training_data['population'])
    population = population(inputs.get('population'))

    # Create buckets for latitude and longitude (in normalized scale)
    latitude_boundaries = np.linspace(-3, 3, 21)
    longitude_boundaries = np.linspace(-3, 3, 21)

    latitude = tf.keras.layers.Discretization(
        bin_boundaries=latitude_boundaries,
        name='discretized_latitude')(latitude)
    
    longitude = tf.keras.layers.Discretization(
        bin_boundaries=longitude_boundaries,
        name='discretized_longitude')(longitude)

    # Feature cross
    feature_cross = tf.keras.layers.HashedCrossing(
        num_bins = len(latitude_boundaries) * len(longitude_boundaries),
        output_mode='one_hot',
        name='cross_lat_lon')([latitude, longitude])

    # Concatenate all features
    all_features = tf.keras.layers.Concatenate(name='all_features')(
        [feature_cross, median_income, population])

    ## Get a first approximation with a linear regression model
    
    # Linear regression output
    label_name = 'median_house_value'

    outputs = linear_regression_output(all_features)

    # Build the model
    model = build_model(inputs, outputs, learning_rate)

    # Train the model
    epochs, mse, history = train_model(model, training_data, n_epochs, 
                                batch_size, label_name, validation_split)
    
    # Plot the loss curve
    plot_loss_curve(epochs, mse, history['val_mean_squared_error'])

    # Evaluate the model
    x_test = {name:np.array(value) for name, value in testing_data.items()}

    # Normalize the labels
    median_house_value = tf.keras.layers.Normalization(axis=None)
    median_house_value.adapt(np.array(testing_data['median_house_value']))
    y_test = median_house_value(np.array(x_test.pop(label_name)))

    print("\nEvaluate the model on the test set:")
    model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

    ## Get a better approximation with a DNN model

    # DNN output
    dnn_outputs = dnn_output(all_features)
    # You can also modify the number of units in the hidden layers,
    # for example, dnn_outputs = dnn_output(all_features, units=(10, 6))
    # seems to give a better result.

    # Build the model
    model = None
    model = build_model(inputs, dnn_outputs, learning_rate)

    # Train the model
    epochs, mse, history = train_model(model, training_data, n_epochs, 
                                batch_size, label_name, validation_split)

    # Plot the loss curve
    plot_loss_curve(epochs, mse, history['val_mean_squared_error'])

    # Evaluate the model
    print("\nEvaluate the model on the test set:")
    model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

if __name__ == "__main__":
    # Play with hyperparameters to adjust the model.
    lr = 0.01
    epochs = 20
    bs = 1000
    val_split = 0.2

    main(lr, epochs, bs, val_split)