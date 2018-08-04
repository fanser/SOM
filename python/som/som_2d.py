from sampler import sample_by_batch
from initilization import w_init_from_data, TopoExpWeight, LRExpDecay, w_init_random
from matcher import l2_dist_matcher

import numpy as np


class SOM2D():
    def __init__(self, R0, num_epoch, lr0):
        self.topo_func = TopoExpWeight(R0, num_epoch)
        self.matcher = l2_dist_matcher
        self.num_epoch = num_epoch
        self.lr0 = lr0
        #self.matcher = cosine_matcher

    def calc_delta_W(self, topo_w, X, W, winner):
        '''
        Input:
            topo_w: m*n matrix the topographic weight 
            X: n*c float array the feature vector
            W: h*w*c float mat, the parameters of SOM 
            winner: the winner index which is expand by row
        '''
        map_h, map_w = W.shape[0], W.shape[1]
        delta_w = np.zeros_like(W)
        count_map = np.ones((map_h, map_w, 1))

        center_h = (topo_w.shape[0] -1 )/2
        center_w = (topo_w.shape[1] - 1)/2
        for x, idx in zip(X, winner):
            h, w = idx / map_w, idx % map_w
            up, down = h - center_h, h + center_h + 1
            c_up, c_down = 0, 2*center_h + 1
            if up < 0:
                up, c_up = 0, center_h - h
            if down > map_h:
                down, c_down = map_h, center_h  + map_h - h
            left, right = w - center_w, w + center_w +1
            c_left, c_right = 0, 2*center_w + 1

            if left < 0:
                left, c_left = 0, center_w - w
            if right > map_w:
                right, c_right = map_w, center_w  + map_w - w
            diff = x.reshape(1, 1, -1) - W[up:down, left:right, :]
            diff *= topo_w[c_up:c_down, c_left:c_right][:, :, np.newaxis]
            delta_w[up:down, left:right, :] += diff
            count_map[up:down, left:right, :] += 1
        return delta_w /count_map
        #return delta_w


    def train(self, X, W, num_epoch, batch_size):
        assert X.shape[-1] == W.shape[-1]
        num_iters = int(np.ceil(float(X.shape[0]) / batch_size))
        for epoch in range(num_epoch):
            idxs = np.arange(X.shape[0])
            np.random.shuffle(idxs)
            X_shuffled = X[idxs, :]
            topo_w = self.topo_func.get_weight(t=epoch)
            lr = self.lr0 * np.exp(-float(epoch)/self.num_epoch)
            for iter in range(num_iters):
                X_batch = X_shuffled[iter*batch_size : min(X.shape[0], (iter+1)*batch_size)]
                #X_batch = X[np.random.choice(X.shape[0], 1), :]
                winner = self.matcher(X_batch, W.reshape(-1, W.shape[-1]))
                #topo_w = np.array([[0,0,0],[0, 1, 0],[0,0,0]])
                delta_w = self.calc_delta_W(topo_w, X_batch, W, winner)
                delta_w *= lr
                W += delta_w
            print 'Epoch {}/{}'.format(epoch, num_epoch)

def Normalize(X):
    '''
    Default axis is -1.
    Input:
        X : (...,n , c) float n-d array
    Output:
        X_normed: same shape with X
    '''
    assert len(X.shape) > 1
    c = X.shape[-1]
    X_reshape = X.reshape(-1, c)
    X_normed = X_reshape / np.sqrt(np.sum(X_reshape**2, 1, keepdims=True)) + 1e-10
    return X_normed.reshape(X.shape)

if __name__ == "__main__":
    import numpy as np
    num_dim = 400
    R0 = num_dim / 2.0
    num_iters = 200 
    batch_size = 4
    lr0 = 0.1
    W = w_init_random(num_dim*num_dim, 3).reshape(num_dim, num_dim, -1)
    X =  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                   [0, 0.5, 0.25], [0, 0, 0.5], [1, 1, 0.2],
                   [1, 0.4, 0.25], [1, 0, 1]]
                   )
    som2d = SOM2D(R0, num_iters, lr0)
    som2d.train(X, W, num_iters, batch_size)
    idxs = l2_dist_matcher(W.reshape(-1, 3), X)

    import cv2
    img = np.zeros((num_dim, num_dim, 3), dtype=np.uint8)
    rgb = X[idxs.reshape(-1), :][:, ::-1].reshape(num_dim, num_dim, 3) * 255
    rgb = rgb.astype(np.uint8)
    img += rgb
    cv2.imwrite("./som.jpg", img)
