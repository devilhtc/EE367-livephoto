
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth




def get_all_seg(img_id,addcoor=False,ratio=2,up=0,outpath='data/results/seg_all/'):
    oimg=cv2.imread('data/original/'+img_util.num2strlen2(img_id)+'.JPG')
    vid=img_util.read_in_video('data/original/'+img_util.num2strlen2(img_id)+'.MOV')
    # find out which frame correspond to image
    fit=img_util.find_fit(vid,oimg)

    d2=np.shape(vid[0])

    # resize - make it smaller so that segmentation runs faster
    newsize=(int(d2[1]/ratio),int(d2[0]/ratio))
    prevf=cv2.resize(vid[fit],newsize)
    nextf=cv2.resize(vid[fit+1],newsize)
    flow=img_util.calc_deepflow(prevf,nextf)
    flow_rgb=img_util.flow2rgb(flow)
    # resize
    rimg=cv2.resize(oimg,newsize)
    rimg=cv2.cvtColor(rimg,cv2.COLOR_RGB2LUV)
    img=np.concatenate((rimg ,flow),2)
    n_clusters0 ,label_image_ms0=img_util.segment_image_ms(rimg,up,addcoor)

    n_clusters ,label_image_ms=img_util.segment_image_ms(img,up,addcoor)

    # plotting and saving figure

    fig = plt.figure()

    subfig1 = fig.add_subplot(221)
    subfig1.set_aspect('auto')
    subfig1.imshow(oimg)
    subfig2 = fig.add_subplot(222)
    subfig2.set_aspect('auto')
    subfig2.imshow(flow_rgb)
    subfig3 = fig.add_subplot(223)
    subfig3.set_aspect('auto')
    subfig3.imshow(label_image_ms0)
    subfig4 = fig.add_subplot(224)
    subfig4.set_aspect('auto')
    subfig4.imshow(label_image_ms)
    fig.savefig(outpath+img_util.num2strlen2(img_id)+'.png')

def main():
    get_all_seg(5,addcoor=True)

if __name__ == '__main__':
    main()
