# Playground

import sys
import cv2
import numpy as np
import time
import utils.img_util as img_util
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs


def clustering_ms():
    centers = [[1, 1, 1], [-1, -1, -2], [1, -1, -3]]
    X, _ = make_blobs(n_samples=20, centers=centers, cluster_std=0.6)
    print(np.shape(X))
    print(X)
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    print(labels)
    print(type(labels))
    cluster_centers = ms.cluster_centers_


    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def trial1():
    a=np.random.rand(2,3,4)
    print(a[:,:,0])
    b=img_util.flatten_to_standard_rows(a)
    print(b[:,0])

def trial2():
    a=cv2.imread('data/original/01.JPG')
    # bilaterial filter,
    # inputs: image, window size, color sigma, spatial sigma
    a=cv2.bilateralFilter(a,8,10,3)
    print(np.shape(a))
    d=np.shape(a)

    ratio=2
    a=cv2.resize(a,(int(d[0]/ratio),int(d[1]/ratio)),0,0,interpolation=INTER_AREA)
    l,b=img_util.segment_image(a)
    print("number of estimated clusters : %d" % l)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(b)

    ax.set_aspect('auto')
    ax.imshow(b, interpolation='nearest',cmap='hot')
    ax.set_aspect('auto')
    fig.savefig('auto.png')


def trial3():
    a=np.random.rand(300,500,2)
    c=img_util.add_coordinates(a)
    print(np.shape(c))

def trial4():
    img=cv2.imread(filename)

def main():
    #clustering_ms()
    #trial1()
    trial2()

if __name__=="__main__":
    main()
