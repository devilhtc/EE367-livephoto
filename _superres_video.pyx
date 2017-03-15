cimport numpy as np
import numpy as np
import cython
import cv2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def enhance_video_adding(np.ndarray[unsigned char, ndim=3] high_res_image, np.ndarray[unsigned char, ndim=3] low_res_image_resized, np.ndarray[float, ndim=3] of):
    cdef np.ndarray[unsigned char, ndim=3] low_res_image_enhanced = low_res_image_resized.copy()
    cdef int new_x
    cdef int new_y
    cdef np.ndarray[unsigned char, ndim=1] new_pixel
    cdef np.ndarray[short, ndim=3] high_res_texture = high_res_image.astype('i2') - cv2.bilateralFilter(high_res_image, 5, 2.8 * 4, 2.8).astype('i2')
    for x, y in np.ndindex(low_res_image_enhanced.shape[0], low_res_image_enhanced.shape[1]):
        new_x = x + int(of[x, y][1])
        new_y = y + int(of[x, y][0])
        if 0 <= new_x < low_res_image_enhanced.shape[0] and 0 <= new_y < low_res_image_enhanced.shape[1]:
            new_pixel = high_res_image[new_x, new_y]
            low_res_image_enhanced[x, y] = np.clip(low_res_image_enhanced[x, y].astype('i2') +
                                                   high_res_texture[new_x, new_y] * 10, 0, 255).astype('u1')
    return low_res_image_enhanced

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def enhance_video_mapping(np.ndarray[unsigned char, ndim=3] high_res_image, np.ndarray[unsigned char, ndim=3] low_res_image_resized, np.ndarray[float, ndim=3] of):
    cdef np.ndarray[unsigned char, ndim=3] low_res_image_enhanced = low_res_image_resized.copy()
    cdef int new_x
    cdef int new_y
    cdef np.ndarray[unsigned char, ndim=1] new_pixel
    for x, y in np.ndindex(low_res_image_enhanced.shape[0], low_res_image_enhanced.shape[1]):
        new_x = x + int(of[x, y][1])
        new_y = y + int(of[x, y][0])
        if 0 <= new_x < low_res_image_enhanced.shape[0] and 0 <= new_y < low_res_image_enhanced.shape[1]:
            new_pixel = high_res_image[new_x, new_y]
            if np.sum(np.abs(new_pixel.astype('i2') - low_res_image_enhanced[x, y].astype('i2'))) < 200:
                low_res_image_enhanced[x, y] = new_pixel
    return low_res_image_enhanced
