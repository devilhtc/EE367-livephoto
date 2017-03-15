import pyximport

pyximport.install()
from _superres_video import enhance_video_adding, enhance_video_mapping
from LivePhoto import LivePhoto
import cv2


def superres_video_frame(high_res_image, low_res_image_resized, method_name="mapping"):
    frame1 = cv2.cvtColor(low_res_image_resized, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY)
    df = cv2.optflow.createOptFlow_DeepFlow()
    of = df.calc(frame1, frame2, None)
    if method_name == "adding":
        return enhance_video_adding(high_res_image, low_res_image_resized, of)
    elif method_name == "mapping":
        return enhance_video_mapping(high_res_image, low_res_image_resized, of)
    else:
        return None


def superres_video(photo: LivePhoto, method_name="mapping"):
    high_res_image = photo.image
    enhanced_frames = []
    for frame in photo.video:
        frame_resized = cv2.resize(frame, (4032, 3024))
        frame_enhanced = superres_video_frame(high_res_image, frame_resized, method_name)
        enhanced_frames += [frame_enhanced]
        print("next")
    return enhanced_frames


if __name__ == "__main__":
    import numpy as np
    from skimage import io
    import sys


    def imsave(location, mat):
        if mat.dtype == np.dtype('float'):
            mat = (mat * 256).astype('u1')
        if len(mat.shape) == 3 and mat.shape[2] == 3:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        io.imsave(location, mat)


    photo = LivePhoto("resources/sample_00/IMG_2454")
    high_res_image = photo.image
    frame = photo.video[int(sys.argv[1])]
    frame_resized = cv2.resize(frame, (4032, 3024))
    frame_enhanced = superres_video_frame(high_res_image, frame_resized)
    imsave("data/" + sys.argv[1] + ".png", frame_enhanced)
