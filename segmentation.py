
import cv2
import numpy as np
import utils.img_util as img_util
import matplotlib.pyplot as plt




def get_all_seg(img_id,addcoor=False,ratio=2,up=0,outpath='results/seg_all/'):
    oimg=cv2.imread('resources/sample_01/'+img_util.num2strlen2(img_id)+'.JPG')
    oimg=cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)

    vid=img_util.read_in_video('resources/sample_01/'+img_util.num2strlen2(img_id)+'.MOV')
    # find out which frame correspond to image
    fit=img_util.find_fit(vid,oimg)

    d2=np.shape(vid[0])

    # resize - make it smaller so that segmentation runs faster
    newsize=(int(d2[1]/ratio),int(d2[0]/ratio))
    prevf=cv2.cvtColor(cv2.resize(vid[fit],newsize),cv2.COLOR_BGR2RGB)
    nextf=cv2.cvtColor(cv2.resize(vid[fit+1],newsize),cv2.COLOR_BGR2RGB)
    flow=img_util.calc_deepflow(prevf,nextf)
    flow_rgb=img_util.flow2rgb(flow)
    # resize
    rimg=cv2.resize(oimg,newsize)
    img=np.concatenate((rimg ,flow),2)
    quantile=0.1
    n_clusters0 ,label_image_ms0=img_util.segment_image_ms(rimg,up,addcoor,quantile)

    n_clusters ,label_image_ms=img_util.segment_image_ms(img,up,addcoor,quantile)
    label_image_ms=cv2.medianBlur(label_image_ms.astype(np.float32),5)
    label_image_ms=cv2.medianBlur(label_image_ms.astype(np.float32),5)
    # plotting and saving figure

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
    subfig3.imshow(label_image_ms0)
    subfig3.set_title('naive segmenation')
    subfig3.axes.get_xaxis().set_visible(False)
    subfig3.axes.get_yaxis().set_visible(False)

    subfig4 = fig.add_subplot(224)
    subfig4.set_aspect('auto')
    subfig4.imshow(label_image_ms)
    subfig4.set_title('segmenation w/ flow')
    subfig4.axes.get_xaxis().set_visible(False)
    subfig4.axes.get_yaxis().set_visible(False)

    fig.savefig(outpath+img_util.num2strlen2(img_id)+'.png')

def main():

    get_all_seg(12,addcoor=True)

if __name__ == '__main__':
    main()
