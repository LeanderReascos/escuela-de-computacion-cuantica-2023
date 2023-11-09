import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

def scale_data(data):
        # Scale the data from -pi to pi
        scaled_data = data + np.abs(np.min(data)) 
        scaled_data = scaled_data / np.max(scaled_data) * 2 * np.pi  - np.pi
        return scaled_data

class Data:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
    
    def prepare_data(self):
        # Assuming 'Target' is the target variable and other columns are features
        X = self.train_data.drop('Target', axis=1)
        self.y = self.train_data['Target']

        X_test = self.test_data.drop('Target', axis=1)
        self.y_test = self.test_data['Target']

        # Initialize a FunctionTransformer using the defined function
        scaler = FunctionTransformer(scale_data)

        # Transform the data using the defined scaler
        self.X_train_scaled = scaler.fit_transform(X)
        self.X_test_scaled = scaler.transform(X_test)