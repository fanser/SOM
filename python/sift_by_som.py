from som.som_2d import SOM2D
from som.initilization import w_init_random
from som.matcher import l2_dist_matcher
import numpy as np
import os

def genMap(X, W):
    import cv2
    idxs = l2_dist_matcher(X, W.reshape(-1, W.shape[-1]))
    heat_map = np.zeros((W.shape[0], W.shape[1]))
    w, h, _ = W.shape
    for idx in idxs:
        heat_map[idx / w, idx %w] += 1.0
    np.save('./heat_map{}.npy'.format(w), heat_map)
    heat_map /= heat_map.max() + 1e-10
    heat_map = (heat_map * 255).astype(np.uint8)
    cv2.imwrite("./heat_map_{}.jpg".format(w), heat_map)

if __name__ == "__main__":
    feat_dir = "/data01/home/fanzhongyue/disk/video9w/images/sift_100/db"
    sifts = []
    num_sift = 30
    for feat_file in os.listdir(feat_dir)[:5000]:
        feat = np.load(os.path.join(feat_dir, feat_file))
        sift = feat['descirptors'][:num_sift]
        if sift.shape[0] > 0:
            sifts.append(sift)
    sifts = np.vstack(sifts)
    sifts /= 255.0
    #sifts /= np.sqrt(np.sum(sifts**2, axis=1, keepdims=1)) + 1e-10
    H, W = 50, 50
    R0 = max(H, W) /2
    num_epoch = 20
    barch_size = 200
    lr0 = 0.1
    dim = sifts.shape[1]
    weight = w_init_random(H*W, dim).reshape(H, W, dim)
    #som = SOM2D(R0, num_epoch, lr0)
    #som.train(sifts, weight, num_epoch, barch_size)
    map_file = "./mapW_{}.npy".format(H)
    #np.save(map_file, weight)
    weight = np.load(map_file)
    genMap(sifts, weight)
