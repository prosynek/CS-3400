import numpy as np

class Input:
    def predict(self, input_data):
        # append a column of 1s for multiplying by the bias / intercept
        constant = np.ones((input_data.shape[0], 1))
        X = np.hstack([constant, input_data])
        return X

class Neuron:
    def __init__(self, input_layers, weights):
        self.weights = weights
        self.input_layers = input_layers

    def predict(self, input_data):
        predictions = []
        for layer in self.input_layers:
            predictions.append(layer.predict(input_data))
        result = np.dot(np.hstack(predictions), self.weights)
        zeros = np.zeros(result.shape)
        return np.fmax(result, 0.0)
    
class HStack:
    def __init__(self, input_layers):
        self.input_layers = input_layers
        
    def predict(self, input_data):
        predictions = []
        for layer in self.input_layers:
            predictions.append(layer.predict(input_data).reshape(-1, 1))
        return np.hstack(predictions)