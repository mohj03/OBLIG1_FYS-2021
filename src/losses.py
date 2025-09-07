import numpy as np

def cross_entropy(y, y_hat, eps):
    y_hat = np.clip(y_hat, eps, 1-eps) # to avoid log(0)
    loss = - (y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)) # cross-entropy loss
    return np.mean(loss)