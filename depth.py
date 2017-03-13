
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt

def get_all_dep(img_id,offset=1,outpath='data/results/dep_all/'):
    oimg=cv2.imread('data/original/'+img_util.num2strlen2(img_id)+'.JPG')
    vid=img_util.read_in_video('data/original/'+img_util.num2strlen2(img_id)+'.MOV')
    # find out which frame correspond to image
    fit=img_util.find_fit(vid,oimg)
    framel=vid[fit]
    framer=vid[fit-offset]
    disparity=get_disparity(framel,framer)
    #cv2.imshow('left frame',framel)
    plt.imshow(cv2.cvtColor(framel,cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(framer,cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(disparity,'gray')
    plt.show()

    # save fig
    fig = plt.figure()
    subfig1 = fig.add_subplot(121)
    subfig1.set_aspect('auto')
    subfig1.imshow(cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB))
    subfig2 = fig.add_subplot(122)
    subfig2.set_aspect('auto')
    subfig2.imshow(disparity,'gray')
    fig.savefig(outpath+img_util.num2strlen2(img_id)+'.png')


def get_disparity(framel,framer,numDisparity=16,SADwinSize=15):

    framel=cv2.cvtColor(cv2.cvtColor(framel,cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2GRAY)
    framer=cv2.cvtColor(cv2.cvtColor(framer,cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2GRAY)
    d=np.shape(framel)
    ratio=4
    framel=cv2.resize(framel,(int(d[1]/ratio),int(d[0]/ratio)))
    framer=cv2.resize(framer,(int(d[1]/ratio),int(d[0]/ratio)))
    stereo=cv2.StereoBM_create(numDisparity,SADwinSize)
    disparity = stereo.compute(framel, framer)
    return disparity

def main():
    get_all_dep(4,offset=3)

if __name__ == '__main__':
    main()
