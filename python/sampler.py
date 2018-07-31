import numpy as np

def sample_by_batch(X, batch_size):
    batches = []
    for i in range(0, X.shape[0], batch_size):
        batches.append(X[i: i+batch_size, :])
    return batches

def sample_by_one(X):
    samples = []
    for x in X:
        samples.append(x.reshape(1, -1))
    return samples

def sample_by_all(X):
    return X
