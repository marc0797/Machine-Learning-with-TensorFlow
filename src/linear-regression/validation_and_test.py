## Linear regression on real dataset

# In this case, we will be using validation and test sets
# to evaluate the model's performance, and determine
# whether the model is overfitting the training data.

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Define a function that builds an empty model.
def build_model(learning_rate):
    """ Create and compile a simple linear regression model. """    
    # Most simple tf.keras model is sequential
    # Topography of the model is a single node in a single layer.
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(units=1)
    ])

    # Compile the model and configure training to minimize MSE.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

# Function which trains the model from the examples you pass.
def train_model(model, data, feature, label, epochs, batch_size, validation_split):
    """ Train the model with the data. """

    # Feed the feature values and the label values to the model.
    history = model.fit(x=data[feature],
                        y=data[label],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # List of epochs
    epochs = history.epoch

    # Gather the history of each epoch.
    hist = pd.DataFrame(history.history)

    # Gather root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history

# Function to plot the feature values vs. label values.
def plot_model(trained_weight, trained_bias, data, feature, label):
    """ Plot the trained model against a subset of the training data. """

    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from a subset of the data
    sample = data.sample(n=200)

    # Plot the feature values vs. label values.
    plt.scatter(sample[feature], sample[label], c="b", label="Data", alpha=0.5)

    # Plot a red line representing the model.
    x0 = 0
    y0 = trained_bias
    x1 = sample[feature].max()
    y1 = trained_weight * x1 + trained_bias
    plt.plot([x0, x1], [y0, y1], c='r', label="Model")

    # Render the scatter plot and the red line.
    plt.legend()
    plt.show()

def plot_loss_curve(epochs, error_training, error_validation):
    """ Plot the loss curve, which shows loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    # Skip the first epoch to give the graph a cleaner look.
    plt.plot(epochs[1:], error_training[1:], label="Training Loss")
    plt.plot(epochs[1:], error_validation[1:], label="Validation Loss")
    plt.legend()

    merged_error = error_training[1:] + error_validation[1:]
    y_max = max(merged_error)
    y_min = min(merged_error)
    y_diff = y_max - y_min

    plt.ylim(y_min - y_diff*0.05, y_max + y_diff*0.05)
    plt.show()

# Function to make predictions
def predict(n, data, feature, label, model):
    """ Predict house values based on a feature."""

    batch = data[feature][10000:10000 + n]
    predicted_values = model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousands   in thousands")
    print("--------------------------------------")
    for i in range(n):
        print ("%5.0f %6.0f %15.0f" % (data[feature][10000 + i],
                                       data[label][10000 + i],
                                       predicted_values[i][0]))

def reset(lr, epochs, bs, validation_split, data, feature, label):
    model = None
    model = build_model(lr)
    epochs, rmse, history = train_model(model, data, feature,
                                        label, epochs, bs, validation_split)
    
    plot_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])

    return model

def main(n_epochs=10, lr=0.01, bs=10, val_split=0.2):
    # Load the dataset
    training_data_path = 'data/california_housing_train.csv'
    training_data = pd.read_csv(training_data_path)
    test_data_path = 'data/california_housing_test.csv'
    test_data = pd.read_csv(test_data_path)

    scaling_factor = 1000.0

    # Scale the training and test data.
    training_data["median_house_value"] /= scaling_factor
    test_data["median_house_value"] /= scaling_factor

    # Split the training data into a training and validation set.
    validation_split = val_split

    # Choose features and labels
    feature = "median_income"
    label = "median_house_value"

    # Training the model as is, it may happen that our training set 
    # data is ordered in a specific way, making it different to the 
    # validation set, no matter the split.
    #
    # In this case, the data is sorted by longitude, and it seems that
    # longitude is somewhat correlated with the label, median house value.
    # To avoid this, we should shuffle the data.

    training_data = training_data.reindex(np.random.permutation(training_data.index))

    # Train the model
    model = reset(lr, n_epochs, bs, validation_split, training_data, feature, label)

    # Evaluate the model
    x_test = test_data[feature]
    y_test = test_data[label]

    results = model.evaluate(x_test, y_test, batch_size=bs)

    # Ideally, the loss on the test set should be similar to the loss 
    # on the validation and training sets.
    # Playing with the hyperparameters, we are able to find a model
    # that performs well on the training and validation sets, 
    # and obtains a similar loss on the test set.

if __name__ == "__main__":
    # Play with hyperparameters to adjust the model.
    epochs = 70
    learning_rate = 0.08
    batch_size = 100

    # Choose the validation split (from 0.0 to 1.0).
    # A larger validation split will give you a more accurate 
    # estimate of the model's performance, but we will have 
    # less data to train the model.
    validation_split = 0.2

    main(n_epochs=epochs, lr=learning_rate, bs=batch_size, 
         val_split=validation_split)