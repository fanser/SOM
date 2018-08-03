import numpy as np

def w_init_from_data(x, num_centers):
    '''
    Sampling data to initialize the centers
    '''
    idxs = np.arange(0, x.shape[0], dtype=np.int32)
    np.random.shuffle(idxs)
    idxs = idxs[:num_centers]
    centers = x[idxs, :]
    return centers

def w_init_random(num_centers, dims):
    centers  = np.random.rand(num_centers, dims)
    #centers /= np.sqrt(np.sum(centers**2, axis=1, keepdims=True)) + 1e-10
    return centers


class TopoExpWeight():
    def __init__(self, R0, num_iters):
        self.R0 = R0
        self.num_iters = num_iters
        self.T0 = num_iters / np.log(R0+0.001)

    def get_weight(self, t):
        radius = self.R0 * np.exp(-t/self.T0)
        S_sq, mask = self._get_S_square(radius)
        T = np.exp(-S_sq / (2*radius**2))
        T *= mask
        return T

    def _get_S_square(self, radius=2):
        up_radius = int(radius)
        radius_sq = radius **2
        W = 2*up_radius + 1
        S = np.zeros((W, W))
        mask = np.zeros((W, W))
        for h in range(W):
            for w in range(W):
                dist_sq = (h - up_radius) **2 + (w-up_radius)**2
                if dist_sq <= radius_sq:
                    mask[h, w] = 1
                    S[h, w] = dist_sq
        return S, mask


class LRExpDecay():
    def __init__(self, lr0=0.1, t0=1e2):
        self.lr0 = lr0
        self.t0 = t0

    def get_lr(self, t):
        return self.lr0 * np.exp(-t / self.t0)
