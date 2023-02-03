import numpy as np
from scipy.stats import mode

def linear_decision_boundary_classifier(decision_boundary_line_vec, training_points, training_labels, prediction_points):
    """
    decision_boundary_line_vec: Vector representation of a linear decision boundary.  Convert the line to the form Ax + By + ... + C = 0.  Then turn the coefficients into a vector.  So, for example, 5x - y + 2 = 0 will turn into <5, -1, 2>.
    
    training_points: 2D numpy array of points used to train the model.  The number of columns must be one less than the length of the decision boundary vector.
    
    training_labels: training labels as a 1D numpy array.  The length must match the number of rows in training_points.
    
    prediction_points: points to predict labels for as a 1D numpy array.  The number of columns must match the number of columns in training_points.
    """
    
    # add column of 1s to match dimension of decision_boundary_line_vec
    training_points = np.hstack([training_points, np.ones((len(training_points), 1))])
    prediction_points = np.hstack([prediction_points, np.ones((len(prediction_points), 1))])
    
    # calculate dot product. sign of the dot product tells us which "side" of the
    # line on which the points are located
    training_dot_products = np.dot(training_points, decision_boundary_line_vec)
    
    # find the most common label on the negative side
    # note that "mode" returns a 2-tuple of the most common
    # value and its count so we grab the first item
    neg_mask = training_dot_products < 0.0
    neg_label = mode(training_labels[neg_mask])[0]
    
    # find the most common label on the positive side
    pos_mask = training_dot_products > 0.0
    pos_label = mode(training_labels[pos_mask])[0]
    
    # find which side of the line the predictions points are located on
    prediction_dot_products = np.dot(prediction_points, decision_boundary_line_vec)
    
    # predict labels based on which side of the line the points are located on
    prediction_labels = np.zeros(len(prediction_points))
    prediction_labels = prediction_labels.astype("object")
    neg_mask = prediction_dot_products <= 0.0
    prediction_labels[neg_mask] = neg_label
    pos_mask = prediction_dot_products > 0.0
    prediction_labels[pos_mask] = pos_label
    
    return prediction_labels