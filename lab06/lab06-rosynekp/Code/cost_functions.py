import numpy as np

class GaussianCostFunction:
    """
    Implements a cost function for fitting a Gaussian (normal) distribution.
    """
    def __init__(self, features, y_true):
        """
        The constructor takes the feature matrix and true y values
        for the training data.
        """
        self.features = features
        self.y_true = y_true
    
    def predict(self, features, params):
        """
        Predicts the y values for each data point
        using the feature matrix and the model parameters.
        
        We expect that the features are a Nx1 matrix of
        x values.  The params is a length-2 array of the
        mean (mu) and std deviation (sigma).
        """
        mu, sigma = params
        return np.power(sigma * np.sqrt(2 * np.pi), -1) * np.exp(-0.5 * ((features - mu) / sigma)**2)
        
    def _mse(self, y_true, pred_y):
        """
        Calculates the mean-squared error between the predicted and
        true y values.
        """
        return np.square(y_true - pred_y).mean()
        
    def cost(self, params):
        """
        Calculates the cost function value for the model's predictions
        using the given params.
        
        This should:
        1. Use the params and data's features to predict the y values
        2. Calculate the error between the true and predicted y values
        3. Return the error
        """
        y_pred = self.predict(self.features, params)
        return self._mse(self.y_true, y_pred)
        
class LinearCostFunction:
    """
    Implements a cost function for a linear regression model.
    """
    def __init__(self, features, y_true):
        """
        The constructor takes the feature matrix and true y values
        for the training data.
        
        """
        self.features = features
        self.y_true = y_true
    
    def predict(self, features, params):
        """
        Predicts the y values for each data point
        using the feature matrix and the model parameters.
        
        We expect that the features are a NxM matrix.
        The params are a 1D array of length M.
        """
        return np.sum(features * params, axis=1)
        
    def _mse(self, y_true, pred_y):
        """
        Calculates the mean-squared error between the predicted and
        true y values.
        """
        return np.square(y_true - pred_y).mean()
        
    def cost(self, params):
        """
        Calculates the cost function value for the model's predictions
        using the given params.
        
        This should:
        1. Use the params and data's features to predict the y values
        2. Calculate the error between the true and predicted y values
        3. Return the error
        """
        y_pred = self.predict(self.features, params)
        return self._mse(self.y_true,  y_pred)