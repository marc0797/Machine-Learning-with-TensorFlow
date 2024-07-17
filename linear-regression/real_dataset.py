## Linear regression on real dataset

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
def train_model(model, data, feature, label, epochs, batch_size=10):
    """ Train the model with the data. """

    # Feed the feature values and the label values to the model.
    history = model.fit(x=data[feature],
                        y=data[label],
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

def plot_loss_curve(epochs, rmse):
    """ Plot the loss curve, which shows loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
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

def reset(lr, epochs, bs, data, feature, label):
    model = None
    model = build_model(lr)
    weight, bias, epochs, rmse = train_model(model, data, feature,
                                             label, epochs, bs)
    
    plot_model(weight, bias, data, feature, label)
    plot_loss_curve(epochs, rmse)

    return model

def main(n_epochs=10, lr=0.01, bs=10, feature="population"):
    # Load the dataset
    dataset_path = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
    training_data = pd.read_csv(dataset_path)

    # Scale the label. We do this so that the default parameters 
    # are in a range that is more typical for a learning problem.
    training_data["median_house_value"] /= 1000.0

    # Print the first rows of the pandas DataFrame.
    print(training_data.head())

    # We can obtain some statics of the dataset by using the describe method.
    print(training_data.describe())

    # Notice how the maximum value of several columns is very high compared to the
    # other values. For instance, from looking at the total_rooms column at values
    # (25%, 50%, 75%), one could expect the maximum value to be around 5000. However,
    # the maximum value is 37937.

    # When the dataset presents anomalies in certain features, it is good to be careful
    # about using them as features. That said, these anomalies can sometimes represent
    # anomalies in the labels, which could make it a powerful feature to use.

    # Also, it is possible to pre-process the raw data in order to make it more suitable.

    # Choose the feature and the label
    feature = "total_rooms" # The total number of rooms in a city block.
    label = "median_house_value" # The median house value for California households.

    # In this case, we will try to predict the median house value solely based on 
    # the total number of rooms.

    # Train the model with the chosen feature
    model = reset(lr, n_epochs, bs, training_data, feature, label)

    # Predict the median house value for 10 examples.
    predict(10, training_data, feature, label, model)

    # Most of the predicted values are significantly different from
    # the label value, so the trained model is not very accurate. It
    # could be the case that the first 10 predictions are outliers,
    # but we should try to train the model with a different feature.

    # Train the model with the chosen feature
    feature = feature

    model = reset(lr, n_epochs, bs, training_data, feature, label)

    # Predict the median house value for 10 examples.
    predict(10, training_data, feature, label, model)

    # We can also try adding a synthetic feature, namely creating
    # a new feature that was not present in the original dataset,
    # and train the model with this new feature.

    # Create a synthetic feature
    training_data["rooms_per_person"] = training_data["total_rooms"] \
                                        / training_data["population"]

    # Train the model with the new feature
    feature = "rooms_per_person"

    model = reset(lr, n_epochs, bs, training_data, feature, label)

    # Predict the median house value for 10 examples.
    predict(10, training_data, feature, label, model)

    # The predictions are more accurate than the previous ones, so
    # the new feature is better for training the model. Even then,
    # the model is still not very accurate.

    # The final step is to identify the best feature to use for 
    # training the model. We can do this by calculating the correlation
    # between the features and the label.

    # If the correlation is close to 1, the feature is a good candidate,
    # as the label rises when the feature rises. If the correlation is
    # close to -1, the feature is also a good candidate, as the label
    # falls when the feature rises. However, if the correlation is close
    # to 0, the feature is not a good candidate, as the feature does not
    # linearly relate to the label.

    # Generate a correlation matrix
    correlation_matrix = training_data.corr()

    # Print the correlation matrix
    print("--------------------------------------")
    print("\n\nCorrelations:")
    print(correlation_matrix["median_house_value"].sort_values())

    # Based on the correlation matrix, the median_income feature is the
    # best feature to use for training the model. We will train the model
    # with this feature.

    # Train the model with the chosen feature
    feature = "median_income"

    model = reset(lr, n_epochs, bs, training_data, feature, label)

    # Predict the median house value for 10 examples.
    predict(10, training_data, feature, label, model)
    




if __name__ == "__main__":
    # Play with hyperparameters to adjust the model.
    epochs = 30
    learning_rate = 0.06
    batch_size = 30

    # Choose the feature from:
    # "longitude", "latitude", "housing_median_age", "total_rooms",
    # "total_bedrooms", "population", "households", "median_income"
    feature = "population"

    main(n_epochs=epochs, lr=learning_rate, bs=batch_size, feature=feature)