import sys
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt


def get_all_dep(filename,offset=1,outpath='results/'):
    # read in image and video frames
    oimg=cv2.imread('resources/'+filename+'.JPG')
    oimg=cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
    vid=img_util.read_in_video('resources/'+filename+'.MOV')

    # find out which frame correspond to image
    fit=img_util.find_fit(vid,oimg)
    framel=vid[fit]
    framer=vid[fit-offset]

    # get disparity between two frames
    disparity=img_util.get_disparity(framel,framer)

    # get edges
    edges=img_util.get_edge(oimg)

    # do edge-aware expansion
    disparity_expanded=img_util.expand(disparity,edges)

    # get frames
    prevf=cv2.cvtColor(vid[fit],cv2.COLOR_BGR2RGB)
    nextf=cv2.cvtColor(vid[fit+1],cv2.COLOR_BGR2RGB)

    # calculate deepflow
    flow=img_util.calc_deepflow(prevf,nextf)
    flow_rgb=img_util.flow2rgb(flow)

    # plot and save figure
    fig = plt.figure()

    subfig1 = fig.add_subplot(221)
    subfig1.set_aspect('auto')
    subfig1.imshow(oimg)
    subfig1.set_title('Original image')
    subfig1.axes.get_xaxis().set_visible(False)
    subfig1.axes.get_yaxis().set_visible(False)

    subfig2 = fig.add_subplot(222)
    subfig2.set_aspect('auto')
    subfig2.imshow(flow_rgb)
    subfig2.set_title('Optical flow')
    subfig2.axes.get_xaxis().set_visible(False)
    subfig2.axes.get_yaxis().set_visible(False)

    subfig3 = fig.add_subplot(223)
    subfig3.set_aspect('auto')
    subfig3.imshow(disparity,'gray')
    subfig3.set_title('Depth map')
    subfig3.axes.get_xaxis().set_visible(False)
    subfig3.axes.get_yaxis().set_visible(False)

    subfig4 = fig.add_subplot(224)
    subfig4.set_aspect('auto')
    subfig4.imshow(disparity_expanded,'gray')
    subfig4.set_title('Improved depth map')
    subfig4.axes.get_xaxis().set_visible(False)
    subfig4.axes.get_yaxis().set_visible(False)

    fig.savefig(outpath+filename+'_result.png')

if __name__ == '__main__':

    filename='for_depth_'
    filename=filename+sys.argv[1]
    # do depth estimation with disparity mapping
    get_all_dep(filename)
