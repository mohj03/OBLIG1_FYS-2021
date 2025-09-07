import numpy as np

def gradient(y, y_hat, X, lambda_, w):
    # Number of training samples (m)
    m = y.shape[0]
    error = y_hat - y  # prediction error
    dw = ((X.T @ error) / m) + (lambda_/m) * w # gradient for weights with ridge
    db = np.sum(error) / m # gradient for bias
    return dw, db
