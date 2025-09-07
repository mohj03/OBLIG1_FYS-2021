import numpy as np

def start_vals(d, seed=42):
    rng = np.random.default_rng(seed) 
    w = rng.normal(0, 1, size=d) # weights initialized from N(0,1)
    b = 0.0
    return w, b

def z_calc(X, w, b):
    z = X @ w + b # linear combination
    return z

def sigmoid(z):
    z_ = np.clip(z, -40, 40) # to avoid overflow
    y_hat = 1 / (1 + np.exp(-z_)) # sigmoid function
    return y_hat
