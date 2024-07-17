## Linear Regression excercise with Synthetic Data

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

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
def train_model(model, feature, label, epochs, batch_size=10):
    """ Train the model with the data. """

    # Feed the feature values and the label values to the model.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # List of epochs
    epochs = history.epoch

    # Gather the history of each epoch.
    hist = pd.DataFrame(history.history)

    # Gather root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

# Function to plot the feature values vs. label values.
def plot_model(trained_weight, trained_bias, feature, label):
    """ Plot the trained model against the training feature and label. """

    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label, c="b", label="Data", alpha=0.5)

    # Plot a red line representing the model.
    x0 = min(feature)
    y0 = trained_weight * x0 + trained_bias
    x1 = max(feature)
    y1 = trained_weight * x1 + trained_bias
    plt.plot([x0, x1], [y0, y1], c='r', label="Model")

    # Render the scatter plot and the red line.
    plt.legend()
    plt.show()

def plot_loss_curve(epochs, rmse):
    """ Plot the loss curve, which shows loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

def main(n_epoch=10, lr=0.01, bs=10):
    # Define the dataset as two random arrays.
    feature = tf.random.normal(shape=[100])
    noise = tf.random.normal(shape=[100])
    label = feature * 3 + 2 + noise

    # Specify the hyperparameters.
    learning_rate = lr
    epochs = n_epoch
    batch_size = bs

    # Build the model.
    my_model = build_model(learning_rate)
    trained_weight, trained_bias, epochs, rmse = \
            train_model(my_model, feature, label, epochs, batch_size)
    
    # Plot the model.
    plot_model(trained_weight, trained_bias, feature, label)
    plot_loss_curve(epochs, rmse)

if __name__ == "__main__":
    # Play with the hyperparameters, to find the ideal model.
    # Generally: Increase the learning rate to train the model faster.
    #            Decrease the learning rate to train the model more accurately.
    #            Increase the number of epochs to train the model for more time.
    #            Decrease the number of epochs to train the model faster.
    #            Increase the batch size to increase efficiency.
    #            Decrease the batch size to update the model's parameters more often.

    learning_rate = 0.05
    epochs = 20
    batch_size = 20

    main(n_epoch=epochs, lr=learning_rate, bs=batch_size)