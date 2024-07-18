# Machine Learning with TensorFlow
Short introduction to machine learning using TensorFlow by Google:

https://developers.google.com/machine-learning/crash-course

This repository contains exercises and projects for learning machine learning
using TensorFlow.

## Project Structure

- 'src/': Source code for the exercises.
- 'data/': Datasets used for the exercises.
- 'notebooks/': Jupyter notebooks and notes.
- 'requirements.txt': List of Python packages required for the project.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/marc0797/Machine-Learning-with-TensorFlow.git
    cd Machine-Learning-with-TensorFlow
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## data folder:

- **California housing dataset**: Contains data drawn from the 1990 U.S. Census.
Run `pandas.DataFrame.describe()` to see descriptions, data ranges and types for 
each feature in the set.

## src folder:
### Pandas folder:
Contains a short tutorial on how to use the Python library Pandas to
create and make use of DataFrames.

### Linear regression folder:
Contains some short excercises about using Linear Regression decision models
to make predictions according to some feature in the data.

### Logistic regression folder:
Contains a short exercise with a binary classification model to try to predict
whether the median value of the houses in a city is high from some features 
provided by the data.