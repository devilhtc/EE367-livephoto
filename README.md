# Final project of stanford EE-367 winter 2017 #
## Team member: th7, jackiey ##

Enviroment and package required:
Python 3.6, OpenCV 3.2.0 (with community contribution), NumPy, SciPy, `scikit-learn`, Cython

Note: The code has only be tested on `Darwin MacBook-Pro.local 16.4.0 Darwin Kernel Version 16.4.0: Thu Dec 22 22:53:21 PST 2016; root:xnu-3789.41.3~3/RELEASE_X86_64 x86_64`

Setting up the enviroment:

```bash
brew install python3
brew install opencv3 --with-python3 --with-contrib --with-nonfree
pip3 install numpy scipy sklearn cython
```

Run the demo:

```bash
export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS" # A bug in Cython preventing us from adding include path in code.
python3 ./demo.py
```
