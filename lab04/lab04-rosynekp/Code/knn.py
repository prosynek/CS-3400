import random

import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """
    def __init__(self, k, aggregation_function):
        """
        Takes two parameters.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. The
        aggregation_function is either "mode" for classification or
        "average" for regression.
        
        Parameters
        ----------
        k : int
           Number of neighbors
        
        aggregation_function : {"mode", "average"}
           "mode" : for classification
           "average" : for regression.
        """
        self.k = k
        if aggregation_function == 'mode' or aggregation_function == 'average':
            self.aggregation_function = aggregation_function
    
          
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        
        Parameters
        ----------
        X : 2D-array of shape (n_samples, n_features) 
            Training/Reference data.
        y : 1D-array of shape (n_samples,) 
            Target values.
        """
        self.X = X
        self.y = y
        

    def predict(self, X):
        """
        Predicts the output variable's values for the query points X.
        
        Parameters
        ----------
        X : 2D-array of shape (n_queries, n_features)
            Test samples.
            
        Returns
        -------
        y : 1D-array of shape (n_queries,) 
            Class labels for each query.
        """
        # find distances between query point and each reference point
        dists = spatial.distance.cdist(X, self.X, metric='euclidean')
        # sort by distance & get the indicies of the k-nearest
        k_nearest = np.argsort(dists, axis=1)[:, :self.k]
        # labels of the k_nearest neighbors
        k_labels = self.y[k_nearest]
        # predicted labels
        if self.aggregation_function == 'mode':
            predictions = stats.mode(k_labels, axis=1, keepdims=True).mode.flatten()
        elif self.aggregation_function == 'average':
            predictions = np.asarray(np.mean(k_labels, axis=1))

        return predictions