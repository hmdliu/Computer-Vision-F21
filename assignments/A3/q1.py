"""
Created on Dec 4, 2021
Author: Haoming(Hammond) Liu
Email: hl3797@nyu.edu
"""

import cv2
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

def get_sift_kp_des(img):
    # handle OpenCV version differences
    try:
        sift = cv2.SIFT_create()
    except:
        sift = cv2.xfeatures2d.SIFT_create()
    # extract SIFT keypoints & descriptors
    return sift.detectAndCompute(img, None)

def matching(des1, des2, thresh=0.9):
    # use cv2.DMatch to display matched pairs later
    match_1st = [cv2.DMatch(i, -1, -1) for i in range(len(des1))]
    match_2nd = [cv2.DMatch(i, -1, -1) for i in range(len(des1))]
    # compute first match & second match
    for i in range(len(des1)):
        for j in range(len(des1)):
            dist = np.linalg.norm(des1[i] - des2[j])
            if match_1st[i].trainIdx == -1:
                match_1st[i].trainIdx = j
                match_1st[i].distance = dist
            elif dist < match_1st[i].distance:
                match_2nd[i].trainIdx = match_1st[i].trainIdx
                match_2nd[i].distance = match_1st[i].distance
                match_1st[i].trainIdx = j
                match_1st[i].distance = dist
            elif dist < match_2nd[i].distance:
                match_2nd[i].trainIdx = j
                match_2nd[i].distance = dist
    # filter matches
    matches = []
    for i in range(len(des1)):
        if match_1st[i].distance < thresh * match_2nd[i].distance:
            matches.append([match_1st[i]])
    return matches            

def merge_and_pad(kp1, kp2, matches):
    # util functions for merging and padding
    cvt1 = lambda tp: np.array([list(tp) + [1]], dtype=np.float).T
    cvt2 = lambda tp: np.array([list(tp) + [1, 0, 0, 0], \
                            [0, 0, 0] + list(tp) + [1]], dtype=np.float)
    cvt3 = lambda tp: np.array([list(tp)], dtype=np.float).T
    # apply cvts to matches
    cor_pairs = []
    pad_matrix = []
    for m in matches:
        src_cor = cvt1(kp1[m[0].queryIdx].pt)   # size: (3, 1)
        src_mat = cvt2(kp1[m[0].queryIdx].pt)   # size: (2, 6)
        dst = cvt3(kp2[m[0].trainIdx].pt)       # size: (2, 1)
        cor_pairs.append((src_cor, dst))
        pad_matrix.append((src_mat, dst))
    return cor_pairs, pad_matrix

def ransac(cor_pairs, pad_matrix, max_iter=100, thresh=10):

    # solve affine transformation params using 3 pairs
    def solve_affine(sample_idx):
        A = np.vstack(tuple(pad_matrix[i][0] for i in sample_idx))
        b = np.vstack(tuple(pad_matrix[i][1] for i in sample_idx))
        try:
            t = np.linalg.solve(A, b)
        except:
            t = None
        return t

    # check inliers within threshold
    def inlier(src, dst):
        return np.linalg.norm(np.matmul(model, src) - dst) < thresh

    # init variables
    best_model = None
    best_inliers = []
    N = len(cor_pairs)
    idx_list = list(range(N))
    # loop for max_iter times
    for i in range(max_iter):
        # sample & solve
        sample_idx = random.sample(idx_list, k=3)
        model = solve_affine(sample_idx)
        if model is None:
            continue
        else:
            model = model.reshape(2, 3)
        # check inliers & stroe the best model
        inliers = [i for i in range(N) if inlier(*cor_pairs[i])]
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers
    return best_model, best_inliers

# solve least square transformation
def get_lstsq_model(pad_matrix, inliers):
    A = np.vstack(tuple(pad_matrix[i][0] for i in inliers))
    b = np.vstack(tuple(pad_matrix[i][1] for i in inliers))
    with warnings.catch_warnings():
        t = np.linalg.lstsq(A, b)[0]
    return t.reshape(2, 3)

# filter useless warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# load images & call SIFT
book_img = cv2.imread('book.pgm', 0)
scene_img = cv2.imread('scene.pgm', 0)
book_kp, book_des = get_sift_kp_des(book_img)
scene_kp, scene_des = get_sift_kp_des(scene_img)
# display SIFT feature points
f = plt.figure(figsize=(12, 8), dpi=80)
f.add_subplot(1, 2, 1)
plt.title('src')
plt.imshow(cv2.drawKeypoints(book_img, book_kp, outImage=None))
f.add_subplot(1, 2, 2)
plt.title('dst')
plt.imshow(cv2.drawKeypoints(scene_img, scene_kp, outImage=None))
plt.waitforbuttonpress()

# macth the keypoints
matches = matching(book_des, scene_des, thresh=0.9)
print('matches size:', len(matches))
plt.imshow(cv2.drawMatchesKnn(book_img, book_kp, scene_img, scene_kp, \
            matches, outImg=None))
plt.waitforbuttonpress()

# solve the transformation using ransac & lstsq
cor_pairs, pad_matrix = merge_and_pad(book_kp, scene_kp, matches)
raw_M, inliers = ransac(cor_pairs, pad_matrix, max_iter=100, thresh=10)
print('M from ransac:')
print(raw_M)
print('inliers size:', len(inliers))
lstsq_M = get_lstsq_model(pad_matrix, inliers)
print('M from lstsq:')
print(lstsq_M)

# display results
f = plt.figure(figsize=(12, 8), dpi=80)
grey_scale = lambda img: cv2.cvtColor(img, 0)
raw_warp = cv2.warpAffine(book_img, raw_M, scene_img.shape[::-1])
lstsq_warp = cv2.warpAffine(book_img, lstsq_M, scene_img.shape[::-1])
f.add_subplot(2, 2, 1)
plt.title('src')
plt.imshow(grey_scale(book_img))
f.add_subplot(2, 2, 2)
plt.title('dst')
plt.imshow(grey_scale(scene_img))
f.add_subplot(2, 2, 3)
plt.title('ransac result')
plt.imshow(grey_scale(raw_warp))
f.add_subplot(2, 2, 4)
plt.title('lstsq result')
plt.imshow(grey_scale(lstsq_warp))
plt.waitforbuttonpress()