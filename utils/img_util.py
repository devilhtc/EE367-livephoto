import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

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
def flatten_to_standard_rows(ximage):
    d=np.shape(ximage)
    image_reshaped=np.reshape(ximage,(d[0]*d[1],d[2]))
    #for i in range(d[2]):
    #    image_reshaped[:,i]=standardize(image_reshaped[:,i])
    return image_reshaped

"""
standardize each row for clustering
"""
def standardize(vector):
    return (vector-np.mean(vector))/np.std(vector)

def upweight_coor(image,ratio):
    upweight=1+ratio/10
    image[:,3]=image[:,3]*upweight
    image[:,4]=image[:,4]*upweight
    return image
"""
segment image (now according to color)

intput: image (x,y,z) np array
output: n_classes
        label_image (x,y) np array
"""
def segment_image(image,method='meanshift'):
    d=np.shape(image)
    #image=add_coordinates(image)
    image_reshaped=flatten_to_standard_rows(image)
    #image_reshaped=upweight_coor(image_reshaped,5)
    bandwidth = estimate_bandwidth(image_reshaped, quantile=0.2, n_samples=4000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image_reshaped)
    labels = ms.labels_
    label_image=np.reshape(labels,(d[0],d[1]))
    labels_unique = np.unique(labels)
    n_clusters= len(labels_unique)
    return n_clusters ,label_image

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
