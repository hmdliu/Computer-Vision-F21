"""
Created on Dec 6, 2021
Author: Haoming(Hammond) Liu
Email: hl3797@nyu.edu
"""

import os
import numpy as np
from scipy.linalg import rq

# check input file & load points
assert os.path.isfile('world.txt')
assert os.path.isfile('image.txt')
world_pts = np.loadtxt('world.txt')
image_pts = np.loadtxt('image.txt')
# print(world_pts.shape, image_pts.shape)

# concatenate ones
N = world_pts.shape[1]
world_pts = np.vstack((world_pts, np.ones((1, N))))
image_pts = np.vstack((image_pts, np.ones((1, N))))
print('world points:', world_pts, sep='\n')
print('image points:', image_pts, sep='\n')

# constracut matrix A
A_rows = []
zero_item = np.zeros((1, 4))
for i in range(N):
    xi, yi, wi = tuple(image_pts.T[i])
    Xi = world_pts.T[i].reshape(1, 4)
    A_rows.append(np.hstack((zero_item, -wi*Xi, yi*Xi)))
    A_rows.append(np.hstack((wi*Xi, zero_item, -xi*Xi)))
A = np.vstack(A_rows)
print('Matrix A: shape =', A.shape)
print(A)

# solve P using SVD
_, s, v = np.linalg.svd(A)
P = v[np.argmin(s)].reshape(3, 4)
print('Matrix P: shape =', P.shape)
print(P)
inhomo = lambda x: x / x[-1]
verify = np.matmul(P, world_pts)
print('verify proj: shape = image_pts =', image_pts.shape)
print(image_pts - inhomo(verify))

# solve C using SVD (null-space of P)
_, s, v = np.linalg.svd(P)
C1 = inhomo(v[-1].reshape(4, 1))[:-1].T.squeeze()
print('Projection Center C: shape =', C1.shape)
print(C1)

# alternative route
r, K = np.linalg.qr(P.T)
R, t = r[:-1].T, r[-1].T
C2 = np.linalg.solve(-R, t)
print('Alternative Route:')
print(C2)




