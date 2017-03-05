# Trial for opencv

import sys
import cv2
import numpy as np
import time


def try_read_in_file(filename,print_image=False,read_color=True):
    if read_color:
        img=cv2.imread(filename)
    else:
        img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("image does not exist")
    else:
        if print_image:
            print(img)
        print('the type of the variable img is')
        print(type(img))
        cv2.imshow('first image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def try_use_cam(max_sec=10,read_color=True):
    cam=cv2.VideoCapture(0)
    start_time = time.time()
    while (time.time()-start_time)<max_sec:
        tf,frame=cam.read()
        if not read_color:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('single frame',frame)
        cv2.waitKey(1)
    cam.release()
    cv2.destroyAllWindows()

def main():
    print("My opencv version is")
    print(cv2.__version__)

    print("Hello, World!")

    # try read in lena.jpg, try printing it
    filename='lena.jpg'
    # try_read_in_file(filename,False,True)

    try_use_cam()

if __name__=='__main__':
    main()
