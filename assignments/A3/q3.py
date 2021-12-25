"""
Created on Dec 6, 2021
Author: Haoming(Hammond) Liu
Email: hl3797@nyu.edu
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def centroid(pts):
    N = pts.shape[1]
    return np.array(
        [np.sum(pts[0]) / N, np.sum(pts[1]) / N], dtype=np.float
    ).reshape(1, 2)

# check input file & load points
assert os.path.isfile('sfm_points.mat')
sfm_pts = sio.loadmat('sfm_points.mat')['image_points']
_, n, m = sfm_pts.shape

# centroids shape: (10, 2)
centroids = np.vstack([centroid(sfm_pts[:, :, i]) 
                            for i in range(m)])
print('t_i for the first camera:\n%s' % centroids[0])

# W shape: (20, 600)
W = np.vstack([sfm_pts[:, :, i] - centroids[i].reshape(2, 1)
                            for i in range(m)])
# print(W)

# U: (20, 20), D: (20, ), VT: (600, 600)
U, D, VT = np.linalg.svd(W)
# M: (20, 3)
M = np.matmul(U[:, :3], np.diag(D[:3]))
# print(M)
print('M_i for the first camera:\n%s' % M[:2])

pts = VT[:3]
print('first 10 world points:\n%s' % pts.T[:10])
ax = plt.axes(projection='3d')
ax.scatter3D(pts[0], pts[1], pts[2])
plt.show()







