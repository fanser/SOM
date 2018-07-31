from sampler import sample_by_batch
from initilization import w_init_from_data, TopoExpWeight, LRExpDecay, w_init_random
from matcher import cosine_matcher, l2_dist_matcher

import numpy as np


class SOM2D():
    def __init__(self, R0, num_iters, lr0):
        self.topo_func = TopoExpWeight(R0, num_iters, lr0)
        self.matcher = l2_dist_matcher
        #self.matcher = cosine_matcher

    def calc_delta_W(self, topo_w, X, W, winner):
        '''
        Input:
            topo_w: m*n matrix the topographic weight 
            X: n*c float array the feature vector (normalized)
            W: h*w*c float mat, the parameters of SOM (normalized)
            winner: the winner index which is expand by row
        '''
        map_h, map_w = W.shape[0], W.shape[1]
        delta_w = np.zeros_like(W)
        count_map = np.ones((map_h, map_w, 1))

        center_h = (topo_w.shape[0] -1 )/2
        center_w = (topo_w.shape[1] - 1)/2
        #print center_h, center_w
        #print idxs
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
            #print "idx is ", idx
            #print "topo bound ", c_up, c_down, c_left, c_right
            #print "map bound ", up, down, left, right
            diff = x.reshape(1, 1, -1) - W[up:down, left:right, :]
            #print "diff shape ", diff.shape, diff.reshape(-1)
            #print x.reshape(1, 1, -1) - W
            #diff = x.reshape(1, 1, -1) - W.reshape(map_h, map_w, -1)
            diff *= topo_w[c_up:c_down, c_left:c_right][:, :, np.newaxis]
            delta_w[up:down, left:right, :] += diff
            count_map[up:down, left:right, :] += 1
            #print X.shape
        return delta_w /count_map
        #return delta_w


    def train(self, X, W, iters, batch_size):
        assert X.shape[-1] == W.shape[-1]
        for iter in range(iters):
            X_batch = X[np.random.choice(X.shape[0], batch_size), :]
            #X_batch = X[np.random.choice(X.shape[0], 1), :]
            winner = self.matcher(X_batch, W.reshape(-1, W.shape[-1]))
            topo_w = self.topo_func.get_weight(t=iter)
            #topo_w = np.array([[0,0,0],[0, 1, 0],[0,0,0]])
            delta_w = self.calc_delta_W(topo_w, X_batch, W, winner)
            W += delta_w 
            #cosine = X.dot(W.reshape(W.shape[1], -1).T)
            #preds = np.argmax(cosine, axis=1)
            #print preds
            #print lr, topo_w
            #print "##########"
            #print delta_w
            #print W
            #print topo_w, X_batch

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
    num_iters = 2000
    lr0 = 0.1
    W = w_init_random(num_dim*num_dim, 3).reshape(num_dim, num_dim, -1)
    X =  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                   [0, 0.5, 0.25], [0, 0, 0.5], [1, 1, 0.2],
                   [1, 0.4, 0.25], [1, 0, 1]]
                   )
    som2d = SOM2D(R0, num_iters, lr0)
    som2d.train(X, W, num_iters, batch_size=4)
    idxs = l2_dist_matcher(W.reshape(-1, 3), X)
    print idxs.reshape(num_dim, num_dim)
    import cv2
    img = np.zeros((num_dim, num_dim, 3), dtype=np.uint8)
    rgb = X[idxs.reshape(-1), :][:, ::-1].reshape(num_dim, num_dim, 3) * 255
    #rgb = X[idxs.reshape(-1), :].reshape(num_dim, num_dim, 3) * 255
    rgb = rgb.astype(np.uint8)
    img += rgb
    print img.shape
    cv2.imwrite("./som.jpg", img)
    '''
    from sklearn import datasets
    num_dim = 10
    R0 = num_dim / 2.0
    num_iters = 2000
    lr0 = 0.1
    iris = datasets.load_iris()
    X = iris.data
    #W  = w_init_from_data(X, 3).reshape(1, 3, -1)
    W  = w_init_random(num_dim, 4).reshape(1, num_dim, -1)
    X = Normalize(X)
    W = Normalize(W)
    som2d = SOM2D(R0, num_iters, lr0)
    print W.reshape(num_dim, 4)
    som2d.train(X, W, num_iters, batch_size=10)
    print W.reshape(num_dim, 4)
    cosine = X.dot(W.reshape(num_dim, 4).T)
    preds = np.argmax(cosine, axis=1)
    labels = iris.target
    print preds
    print labels
    from sklearn.cluster import KMeans
    cluster = KMeans(n_clusters=3)
    cluster.fit(X)
    centers = cluster.cluster_centers_
    cosine = X.dot(centers.T)
    preds2 = np.argmax(cosine, axis=1)
    print preds2
    print centers
    '''

