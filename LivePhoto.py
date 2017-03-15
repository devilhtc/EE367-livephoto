import numpy as np
import cv2

class LivePhoto:
    IMAGE_POSTFIX = ".JPG"
    VIDEO_POSTFIX = ".mov"

    def __init__(self, url):
        self.image = cv2.imread(url + self.IMAGE_POSTFIX)
        cap = cv2.VideoCapture(url + self.VIDEO_POSTFIX)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames += [frame]
        self.video = frames

        scaled_image = cv2.resize(self.image, dsize=(1440, 1080))

        self.key_image_idx = np.argmin([np.sum(np.abs(0.0 + scaled_image - frame))
                                        for frame in self.video])
