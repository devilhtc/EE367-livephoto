import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

"""
add coordinates to image
size (x,y,z)->(x,y,z+2)
"""
def add_coordinates(image):
    d=np.shape(image)
    xs = np.linspace(0, d[0]-1, d[0])
    ys = np.linspace(0, d[1]-1, d[1])
    yv, xv = np.meshgrid(ys, xs)
    xy=np.stack([xv,yv],axis=2)
    return np.concatenate((image,xy),axis=2)

"""
flatten image (x,y,z)
to (x*y,z) and standardize each row
"""
def flatten_to_rows(ximage,s=False):
    d=np.shape(ximage)
    image_reshaped=np.reshape(ximage,(d[0]*d[1],d[2]))
    if s:
        for i in range(d[2]):
            image_reshaped[:,i]=standardize(image_reshaped[:,i])
    return image_reshaped

"""
standardize each row for clustering
"""
def standardize(vector):
    return (vector-np.mean(vector))/np.std(vector)

def upweight_45(image,ratio):
    upweight=1+ratio/10
    image[:,3]=image[:,3]*upweight
    image[:,4]=image[:,4]*upweight
    return image
"""
segment image (with/without coordinates)
with meanshift
intput: image (x,y,z) np array
output: n_classes
        label_image (x,y) np array
"""
def segment_image_ms(image,up=0,addcoor=False):
    d=np.shape(image)
    if addcoor:
        image=add_coordinates(image)
    image_reshaped=flatten_to_rows(image,True)
    if np.shape(image)[2]>=5:
        image_reshaped=upweight_45(image_reshaped,up)
    bandwidth = estimate_bandwidth(image_reshaped, quantile=0.1, n_samples=4000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image_reshaped)
    labels = ms.labels_
    label_image=np.reshape(labels,(d[0],d[1]))
    labels_unique = np.unique(labels)
    n_clusters= len(labels_unique)
    return n_clusters ,label_image

def segment_image_km(image,n_clusters,addcoor=False):
    d=np.shape(image)
    if addcoor:
        image=add_coordinates(image)
    image_reshaped=flatten_to_rows(image,True)
    km=KMeans(n_clusters).fit(image_reshaped)
    labels=km.labels_
    label_image=np.reshape(labels,(d[0],d[1]))
    return label_image

"""
convert integer (0~99)
to str of length 0
padding 0 in the front if input <10
e.g. 0->"00",3->"03",23->"23"
"""
def num2strlen2(i):
    if i<10:
        return '0'+str(i)
    else:
        return str(i)


"""
find out where the picture is in a sequence of frames
inputs:
    frames: all frames of a video
    pic: the high-res picture

outputs:

"""
def find_fit(frames,pic):
    #print(pic)
    oneframe=frames[0]
    d=np.shape(oneframe)
    smallpic=cv2.resize(pic,(d[1],d[0]))
    #print(smallpic)
    diff=[]
    for i in range(len(frames)):
        diff.append(np.sum(np.power(  frames[i]-smallpic,2)  ))
    return np.argmin(diff)

"""
calculate optical flow between two frames
"""
def calc_flow(prevf,nextf):
    prevf=cv2.cvtColor(prevf,cv2.COLOR_RGB2GRAY)
    nextf=cv2.cvtColor(nextf,cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(prevf, nextf, None,0.5, 4, 15, 4, 5, 1.1,1)

"""
calculate optical flow between two frames
"""
def calc_deepflow(prevf,nextf):
    prevf=cv2.cvtColor(prevf,cv2.COLOR_RGB2GRAY)
    nextf=cv2.cvtColor(nextf,cv2.COLOR_RGB2GRAY)
    df=cv2.optflow.createOptFlow_DeepFlow()
    return df.calc(prevf,nextf,None)


"""
make a distribution between 0 and 1
"""
def regularize(a):
    maxa=np.amax(a)
    mina=np.amin(a)
    a=(a-mina)/(maxa-mina)
    return a

"""
convert flow (x,y)
to flow (rgb)
"""
def flow2rgb(flow):
    flow_x=flow[:,:,0]
    flow_y=flow[:,:,1]
    flow_mag, flow_dir=cv2.cartToPolar(flow_x,flow_y)
    extra=np.ones(np.shape(flow_x))
    flow_mag=regularize(flow_mag)
    flow_dir=regularize(flow_dir)
    flow_hsv=np.stack((flow_dir,extra,flow_mag),axis=2)
    flow_rgb=cv2.cvtColor(flow_hsv.astype(np.float32),cv2.COLOR_HSV2BGR)
    flow_rgb=flow_rgb*255
    return flow_rgb.astype('u1')


"""
read in all frames of a video
"""
def read_in_video(vidname):
    vid=cv2.VideoCapture(vidname)

    frames=[]
    while(True):
        ret, frame = vid.read()

        if frame is None:
            break
        if False:
            if np.shape(frame)[0]>1000:
                frame=cv2.flip(frame,0)
                frame=cv2.flip(frame,1)
        frames.append(frame)
    vid.release()
    return frames
