# Given any list of 2D points (Xin, Yin), construct a singly connected nearest-neighbor path in either 'cw' or 'ccw'
# direction
#
# P sets the point to begin looking for the contour from the original ordering of (Xin, Yin), and
# 'direction' sets the direction of the contour

# Usage:
# [Xout, Yout] = points2contour(Xin, Yin, P, direction)
# ex: [Xout, Yout] = points2contour(Xin, Yin, 1, 'cw)
# ex: print(points2contour([1,1,3,2,3], [1,7,3,5,2]), 1)

import numpy as np

def points2contour(Xin, Yin, P = 0, dir = 'cw'):
    if len(Xin) != len(Yin):
        print('Error: length of Xin and Yin should be the same!')
        return -1
    if len(Xin) < 2:
        print('Warning: the point list must have more than two elements')
        return Xin, Yin

    # all points:
    n = len(Xin)
    Ps = np.concatenate((np.array(Xin).reshape(1, n), np.array(Yin).reshape(1, n)),axis = 0) # P is 2xn array

    # build the distance matrix D_{nxn}
    D = np.ones((n, n))*float("inf")
    for i in range(n):
        D[i, :] = np.linalg.norm(Ps - Ps[:, i].reshape(2, 1), axis=0)
        D[i, i] = float('inf')      # prevent self connection

    inds_out = []
    inds_out.append(P)  # starting point
    while(len(inds_out) != n):
        P_pre = P
        P = np.argmin(D[P, :])
        inds_out.append(P)
        D[P_pre, :] = float('inf')
        D[:, P_pre] = float('inf')
        D[P, P_pre] = float('inf')  # set distance of checked pair to inf

        # print('\n')
        # print('P_pre:', P_pre)
        # print('P: ', P)
        # print(D)

    return list(np.array(Xin)[inds_out]), list(np.array(Yin)[inds_out])

# print(points2contour([1,1,3,2,3], [1,7,3,5,2], P = 0))








