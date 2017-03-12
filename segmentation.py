
import os
import sys
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, spectral_clustering
from sklearn.feature_extraction import image


def get_all_seg(img,weights=None):
    n_clusters ,label_image_ms=img_util.segment_image_ms(img)
    print(np.shape(img),np.shape(label_image_ms))
    d=np.shape(label_image_ms)
    plt.imshow(label_image_ms)
    plt.show()
    label_image_ms=np.reshape(label_image_ms,d+(1,))
    img2=np.concatenate((img,label_image_ms),2)

    label_image_km=img_util.segment_image_km(img2,5,True)
    plt.imshow(label_image_km)
    plt.show()

def get_all_seg2(img_id,ratio=2,up=4):
    oimg=cv2.imread('data/original/'+img_util.num2strlen2(img_id)+'.JPG')
    vid=img_util.read_in_video('data/original/'+img_util.num2strlen2(img_id)+'.MOV')
    fit=img_util.find_fit(vid,oimg)
    d2=np.shape(vid[0])
    newsize=(int(d2[1]/ratio),int(d2[0]/ratio))
    prevf=cv2.resize(vid[fit],newsize)
    nextf=cv2.resize(vid[fit+1],newsize)
    flow=img_util.calc_flow(prevf,nextf)

    rimg=cv2.resize(oimg,newsize)
    img=np.concatenate((rimg ,flow),2)
    n_clusters0 ,label_image_ms0=img_util.segment_image_ms(rimg)

    n_clusters ,label_image_ms=img_util.segment_image_ms(img,up)

    plt.imshow(label_image_ms0)
    plt.show()
    plt.imshow(label_image_ms)
    plt.show()


def main():


    get_all_seg2(1)

if __name__ == '__main__':
    main()
