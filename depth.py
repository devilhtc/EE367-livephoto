
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt

def get_all_dep(img_id,offset=1,span=1,outpath='data/results/dep_all/'):
    oimg=cv2.imread('resources/sample_01/'+img_util.num2strlen2(img_id)+'.JPG')
    oimg=cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
    vid=img_util.read_in_video('resources/sample_01/'+img_util.num2strlen2(img_id)+'.MOV')
    # find out which frame correspond to image
    fit=img_util.find_fit(vid,oimg)
    framel=vid[fit]
    framer=vid[fit-offset]
    disparity=get_disparity(framel,framer)
    edges=img_util.get_edge(oimg)
    disparity=img_util.expand(disparity,edges)

    #cv2.imshow('left frame',framel)

    prevf=cv2.cvtColor(vid[fit],cv2.COLOR_BGR2RGB)
    nextf=cv2.cvtColor(vid[fit+1],cv2.COLOR_BGR2RGB)
    flow=img_util.calc_deepflow(prevf,nextf)
    flow_rgb=img_util.flow2rgb(flow)
    # save fig
    fig = plt.figure()
    subfig1 = fig.add_subplot(311)
    subfig1.set_aspect('auto')
    subfig1.imshow(oimg)

    subfig2 = fig.add_subplot(312)
    subfig2.set_aspect('auto')
    subfig2.imshow(flow_rgb)

    subfig3 = fig.add_subplot(313)
    subfig3.set_aspect('auto')
    subfig3.imshow(disparity,'gray')

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
    return cv2.resize(disparity,(d[1],d[0]))

def main():
    j=3
    get_all_dep(j+1)

if __name__ == '__main__':
    main()
