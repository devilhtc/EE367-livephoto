# Playground

import os
import sys
import cv2
import numpy as np
import time
import utils.img_util as img_util
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth, spectral_clustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction import image

def trial10():
    fx=np.array([[1,1,1,-1/2,-1,-1],[1,1,1,-1,-1/2,-1],[1,1,1,-1,-1,-1/2]])
    fy=np.ones(np.shape(fx))
    fx=fx.astype(np.float32)
    fy=fy.astype(np.float32)
    fmag,fdir=cv2.cartToPolar(fx,fy)
    print(fmag,fdir)
    fmag=regularize(fmag)
    fdir=regularize(fdir)
    print(fmag,fdir)

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
        frame=cv2.flip(frame,0)
        frame=cv2.flip(frame,1)
        #print(type(frame))
        #print(np.shape(frame))

        if frame is None:
            break
        frames.append(frame)
    vid2.release()
    return frames

def trial7():
    a=np.random.rand(2,4,3)
    b=[np.random.rand(2,4,3) for i in range(9)]
    b.append(a)
    print(img_util.find_fit( b,a))

def trial6(imgL,imgR):

    stereo = cv2.StereoBM_create(16*3,31)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def trial9(a,b=5,c=3):
    return (a+b)*c



# stereo
def trial8():
    path='data/others/'
    imgR = cv2.imread(path+'Yeuna9x.png',0)
    imgL = cv2.imread(path+'SuXT483.png',0)
    stereo=cv2.StereoBM_create(16,15)
    disparity = stereo.compute(imgL, imgR)

    fig = plt.figure()
    bx = fig.add_subplot(221)
    bx.imshow(imgL)
    bx.set_aspect('auto')
    cx = fig.add_subplot(222)
    cx.imshow(imgR)
    cx.set_aspect('auto')
    fig = plt.figure()
    ax = fig.add_subplot(223)
    ax.imshow(disparity,'gray')
    ax.set_aspect('auto')
    plt.show()


def calc_flow(prevf,nextf):
    prevf=cv2.cvtColor(prevf,cv2.COLOR_BGR2GRAY)
    nextf=cv2.cvtColor(nextf,cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(prevf, nextf, None,0.5, 3, 15, 3, 5, 1,0)

def regularize(a):
    maxa=np.amax(a)
    mina=np.amin(a)
    a=(a-mina)/(maxa-mina)
    return a

def flow2rgb(flow):
    flow_x=flow[:,:,0]
    flow_y=flow[:,:,1]
    flow_y[flow_y==0]=1
    flow_mag, flow_dir=cv2.cartToPolar(flow_x,flow_y)
    extra=np.ones(np.shape(flow_x))
    flow_mag=regularize(flow_mag)
    flow_dir=regularize(flow_dir)
    flow_hsv=np.stack((flow_dir,extra,flow_mag),axis=2)
    flow_rgb=cv2.cvtColor(flow_hsv.astype(np.float32),cv2.COLOR_HSV2RGB)
    flow_rgb=flow_rgb*255
    return flow_rgb.astype('u1')


def main():
    #clustering_ms()
    #trial1()
    #trial2()
    #clustering_sp()
    #trial7()
    #trial8()
    a=4
    print(trial9(a))
    print(trial9(a,c=7))
    if False:
        id=3
        vidname='data/original/'+img_util.num2strlen2(id)+'.MOV'
        imgname='data/original/'+img_util.num2strlen2(id)+'.JPG'
        vid=try_read_in_video(vidname)
        oimg=cv2.imread(imgname)
        #saveframes(id,vid)
        if False:
            l=len(vid)
            print(l)
            offset=3
            middle=img_util.find_fit(vid,oimg)
            middle=int(l/2)
            print(middle)
            img_L=cv2.cvtColor(vid[middle-offset],cv2.COLOR_RGB2GRAY)
            img_R=cv2.cvtColor(vid[middle+offset],cv2.COLOR_RGB2GRAY)
            #img_L=cv2.bilateralFilter(img_L,8,10,3)
            #img_R=cv2.bilateralFilter(img_R,8,10,3)
            trial6(img_L,img_R)
            trial6(img_R,img_L)
        else:
            l=len(vid)
            middle=img_util.find_fit(vid,oimg)
            prevf=vid[middle]
            nextf=vid[middle-1]
            flow=calc_flow(prevf,nextf)
            flow_rgb=flow2rgb(flow)
            plt.imshow(flow_rgb)
            plt.show()

    trial10()

if __name__=="__main__":
    main()
