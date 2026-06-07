import numpy as np

def is_silent(chunk, threshold=0.01):
    return np.mean(np.abs(chunk)) < threshold