## Binary classification using logistic regression

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

def build_model(inputs, learning_rate, METRICS):
    """ Create and compile a classification model """
    # Concatenate the inputs onto a single layer
    concatenated = tf.keras.layers.Concatenate()(inputs.values())
    dense_output = layers.Dense(units=1, name='dense_layer', 
        activation=tf.sigmoid)(concatenated)

    outputs = {
        'dense_layer': dense_output
    }
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Notice that we are using a different loss function for classification
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
    
    return model

def train_model(model, dataset, epochs, label_name, bs):
    """ Train the model by feeding it data """

    # Initialize features and labels
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))

    # Fit the model
    history = model.fit(x=features, y=label, batch_size=bs,
        epochs=epochs, shuffle=True)

    # List of epochs
    epochs = history.epoch

    # Store the history of the training
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
    """ Plot a curve of one or more classification metrics """

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    
    plt.legend()
    plt.show()

def main(lr, epochs, bs, label_name, classification_threshold):
    # Load the dataset
    training_data = pd.read_csv('data/california_housing_train.csv')
    testing_data = pd.read_csv('data/california_housing_test.csv')

    # Shuffle the training data
    training_data = training_data.reindex(np.random.permutation(training_data.index))

    # Convert raw values to their Z-scores
    training_data_mean = training_data.mean()
    training_data_std = training_data.std()
    training_data_norm = (training_data - training_data_mean) / training_data_std

    # Report some z-scores (notice that they are mostly between -2 and +2)
    print("Training Data Summary:")
    print(training_data_norm.head())

    testing_data_norm = (testing_data - training_data_mean) / training_data_std

    # Set a threshold value for binary classification
    # training_data.describe() used to find the 75% percentile value
    # print(training_data.describe())
    threshold = 265000

    # Define a binary label for the classification 0 = false, 1 = true)
    training_data_norm[label_name] = (
        training_data['median_house_value'] > threshold).astype(float)
    testing_data_norm[label_name] = (
        testing_data['median_house_value'] > threshold).astype(float)

    # Print some examples
    print("\nTraining Data Summary:")
    print(training_data_norm.head())

    # Choose the input features
    inputs = {
        'median_income': tf.keras.Input(shape=(1,)),
        'total_rooms': tf.keras.Input(shape=(1,))
    }

    # Define the metrics to use
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy',
            threshold=classification_threshold),
        tf.keras.metrics.Precision(name='precision',
            thresholds=classification_threshold),
        tf.keras.metrics.Recall(name='recall',
            thresholds=classification_threshold)
    ]

    # Uncomment this to see another (better) metric of the model
    METRICS = [
        tf.keras.metrics.AUC(name='auc', num_thresholds=100),
    ]

    # Build the model
    model = build_model(inputs, lr, METRICS)

    # Train the model
    epochs, hist = train_model(model, training_data_norm, epochs, label_name, bs)

    # Plot the metrics
    metric_list = ['accuracy', 'precision', 'recall']
    
    # Also uncomment this
    metric_list = ['auc']

    plot_curve(epochs, hist, metric_list)

    # Evaluate the model
    features = {name: np.array(value) for name, value in testing_data_norm.items()}
    label = np.array(features.pop(label_name))

    print("\nEvaluating the model:")
    model.evaluate(x=features, y=label, batch_size=bs)

if __name__ == '__main__':
    # Tune the hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 100
    label_name = 'median_house_value_is_high'

    # Set the classification threshold.
    #
    #   - If the threshold is set too high, the model will predict fewer positives,
    #     lowering the recall and increasing the precision.
    #   - If the threshold is set too low, the model will predict more positives,
    #     increasing the recall and lowering the precision.
    #
    # Extremal values of the threshold lead to a model with lower accuracies 
    # (always predicting false is 75% accurate in this case).

    classification_threshold = 0.5

    main(lr=learning_rate, epochs=epochs, bs=batch_size, 
        label_name=label_name, classification_threshold=classification_threshold)