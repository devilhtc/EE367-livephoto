# Trial for opencv

import sys
import cv2
import numpy as np
import time

# run file in python 3
# exec(open("./trial.py").read())

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



def calc_flow(prevf,nextf):
    prevf=cv2.cvtColor(prevf,cv2.COLOR_BGR2GRAY)
    nextf=cv2.cvtColor(nextf,cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(prevf, nextf, None,0.5, 3, 15, 3, 5, 1,1)

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

def write_video(source,outname):
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    out = cv2.VideoWriter(outname,fourcc, 20.0, (1080,1440))
    for i in range(len(source)):
        frame=source[i]
        frame = cv2.flip(frame,0)
        out.write(frame)
    out.release()

def try_save_cam():
    cap = cv2.VideoCapture(0) # Capture video from camera

    # Get the width and height of frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    start_time = time.time()
    while (time.time()-start_time)<5:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame,180)
            print(type(frame))
            print(np.shape(frame))
            frame = cv2.flip(frame,0)
            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                break
        else:
            break

    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()



def main():
    #print(sys.path)
    print("My opencv version is")
    print(cv2.__version__)
    #print(cv2.getBuildInformation())

    # try read in lena.jpg, try printing it
    filename='lena.jpg'
    vidname='resources/IMG_2454.mov'
    #try_read_in_file(filename,False,True)
    #try_use_cam()
    if False:
        frames=try_read_in_video(vidname)
        print('here')
        l=len(frames)
        flows=[]
        for i in range(10):
            print(i)
            start_frame=i
            prevf=frames[start_frame]
            nextf=frames[start_frame+1]
            flow0=calc_flow(prevf,nextf)
            flow_rgb=flow2rgb(flow0)
            print(np.shape(flow_rgb))
            print(type(flow_rgb))
            print(flow_rgb)
            cv2.imshow('flow',flow_rgb)
            flows.append(flow_rgb)
        cv2.destroyAllWindows()
        write_video(flows,'out.avi')



if __name__=='__main__':
    main()
