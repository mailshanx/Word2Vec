import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.    
    """

    if x.ndim == 1 \
        or (x.ndim == 2 and x.shape[0] == 1 ) \
        or (x.ndim == 2 and x.shape[1] == 1):
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    elif x.ndim == 2:
        x-= np.max(x, axis = 1).reshape(-1,1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)
    else:
        raise ValueError("can only accept 1 or 2-d inputs to softmax")

    return x


def sigmoid(x):
    """
    Computes the sigmoid function for the input.
    """

    return 1. / (1 + np.exp(-x))


def sigmoid_grad(f):
    """
    Computes the gradient for the sigmoid function 
    """
    return f * (1.0 - f)
