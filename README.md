# Final project of stanford EE-367 winter 2017 #
## Team member: th7, jackiey ##

Enviroment and package required:
Python 3.6, OpenCV 3.2.0 (with community contribution), numpy, scipy, matplotlib, scikit-learn, scikit-image, cython

Note: The code has only be tested on `Darwin MacBook-Pro.local 16.4.0 Darwin Kernel Version 16.4.0: Thu Dec 22 22:53:21 PST 2016; root:xnu-3789.41.3~3/RELEASE_X86_64 x86_64`

Setting up the enviroment:

```bash
brew install python3
brew install opencv3 --with-python3 --with-contrib --with-nonfree
pip3 install numpy scipy matplotlib sklearn cython scikit-image
```

Run the demo:

```bash
export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS" # A bug in Cython preventing us from adding include path in code.

## Segmentation with Optical Flow
python3 segmentation.py 1 # do segementation on the 1st example

## Depth Estimation with Optical Flow
python3 depth.py 1 # do depth estimation on the 1st example

## Super-resolution Video
python3 superres_video.py 1 # do super-resolution with the 1st frame of the live photo
```

For segmentation and depth estimation, two examples inputs and output are supplied.
For super-resolution video, there are 48 frames to select from (ranging from index 0 to 47).
Outputs are stored in the results folder.
