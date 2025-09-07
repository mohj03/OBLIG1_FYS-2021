import numpy as np

def gradient_step(w, b, dw, db, lr, beta, vw, vb, momentum):

    v_w = (lr * dw + beta * vw) # +heavy ball momentum
    v_b = (lr * db + beta * vb) # +heavy ball momentum

    new_w = w - v_w # update weights
    new_b = b - v_b

    return new_w, new_b, v_w, v_b # return updated weights and momentum terms
 