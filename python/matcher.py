import numpy as np

def cosine_matcher(X, W):
   '''
    X and W all need normalized
    Input:
        X: n*c float mat, the normalized data
        W: num_centers*c the parameters of SOM algorithm
   '''
   #X /= np.sqrt(np.sum(X**2, axis =1, keepdims=True)) + 1e-20
   #W /= np.sqrt(np.sum(W**2, axis =1, keepdims=True)) + 1e-20
   cosine = X.dot(W.T)
   indexs = np.argmax(cosine, axis=1)
   '''
   print cosine
   print (X**2).sum()
   print (W**2).sum()
   '''
   return indexs


def l2_dist_matcher(X, W):
    '''
    X and W has same dim
    X: n*c float array
    '''
    assert X.shape[-1] == W.shape[-1]
    X_reshape = X[:, np.newaxis, :]
    W_reshape = W[np.newaxis, :, :]
    dists = X_reshape - W_reshape
    dists = np.sum(dists**2, axis=-1)
    idxs = np.argmin(dists, axis=1)
    return idxs

