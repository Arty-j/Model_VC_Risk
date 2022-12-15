# Model_VC_Risk
This project shows the application of deep neural networks on venture capital investments. The goal is to use a VC firm's past data on sucessful and unsuccessful start-ups to fit a deep neural network model to their data and predict whether a future investment opportunity would be successful or not.

# Technologies
Python implementation: CPython Python version : 3.7.13 

IPython version : 7.31.1


# Libraries and Modules
import pandas as pd

from pathlib import Path

import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,OneHotEncoder

# Project Steps
1. Prepare the data for use on a neural network model

2. Compile and evaluate a binary classification model using a neural network

3. Optimize the neural network model using a variety of modifications

## Step 1: Prepare the Data
- The data is downloaded from `applicants_data.csv` file located in the Resources folder

- All unnecessary data columns are dropped, such as 'EIN' and "NAME"

- All of the data is converted to integers for modeling
   -- all classification data is transformed using the `OneHotEncoder` function from sklearn library: transforming all of the columns containing data type objects into new colunms with binary codes
    
- Data is concatonated into one large dataframe before being split into target data 'y' and feature data 'X'

- Feature data is scaled using `StandardScalar` function: all X data scaled to closer approximate scale of y data

## Step 2: Compile and Evaluate a classification model
- A deep neural network is created by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras

```
# Create the Sequential model instance
nn = Sequential()
# Add the hidden layer with the number of inputs , and the activation function
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))
```

- Model is compiled and fit using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric

- `Evaluate` is run on the model using the test data to determine the model’s loss and accuracy

- Finally the model is saved and exported to an HDF5 file, named "AlphabetSoup.h5"
    
## Step 3: Optimize the model        
All actions from Step 2 are repeated with one change in each model to check for optimization, and improved metrics of Accuracy and Loss

- Two additional models are run with one parameter changed in each

    --A1: the use of the 'tahn' activation function used on the hidden layers
        --- hidden layers remain at 2, epoch remains at 50
        
    
    --A2: 3rd additional hidden layer of nodes is added
        --- activation function is returned to 'relu', epoch remains at 50


# Results
Accuracy and loss remained almost the same across all 3 model runs.  Changing the activation function didn't make a difference in the evaluation results, nor did adding another hidden layer of nodes.  More trials would need to be run on the data, perhaps with fewer features, to gain an accuracy of higher than 74%.

