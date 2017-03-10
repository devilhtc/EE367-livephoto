# Playground

import os
import sys
import cv2
import numpy as np
import time
import utils.img_util as img_util
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

def clustering_sp():
    a=cv2.imread('data/original/01.JPG')
    # bilaterial filter,
    # inputs: image, window size, color sigma, spatial sigma
    a=cv2.bilateralFilter(a,8,10,3)
    #print(np.shape(a))
    d=np.shape(a)
    ratio=20
    newsize=(int(d[1]/ratio),int(d[0]/ratio))
    img=cv2.resize(a,newsize)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    graph = image.img_to_graph(img)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')


    labels_img=np.reshape(labels,(newsize[1],newsize[0]))

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_aspect('auto')
    ax.imshow(img)
    ay = fig.add_subplot(122)
    ay.set_aspect('auto')
    ay.imshow(labels_img, interpolation='nearest',cmap='hot')
    #plt.matshow(labels)

    fig.savefig('sp.png')

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
    a=cv2.resize(a,(int(d[0]/ratio),int(d[1]/ratio)))
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

def trial5():
    # try save array as csv file and read it
    size=(4,5,2)
    b=np.random.rand(4,5,2)
    b.tofile('foo.csv',sep=',',format='%10.8f')
    c = np.reshape(np.genfromtxt('foo.csv', delimiter=','),size)
    print(b)
    print(c)

def trial6():
    directory='data/results/frames/01'
    if not os.path.exists(directory):
        os.makedirs(directory)

def saveframes(id,vid):
    directory='data/results/frames/'+img_util.num2strlen2(id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(len(vid)):
        cv2.imwrite( directory+'/'+img_util.num2strlen2(i)+'.png',vid[i])

def try_read_in_video(vidname):
    vid=cv2.VideoCapture(vidname)
    vid2=vid
    frames=[]
    while(True):
        ret, frame = vid.read()
        #print(type(frame))
        #print(np.shape(frame))

        if frame is None:
            break
        frames.append(frame)
    vid2.release()
    return frames

def main():
    #clustering_ms()
    #trial1()
    #trial2()
    #clustering_sp()
    id=2
    vidname='data/original/'+img_util.num2strlen2(id)+'.MOV'
    vid=try_read_in_video(vidname)
    saveframes(id,vid)

if __name__=="__main__":
    main()
