import cv2
from moviepy.editor import VideoFileClip
clip = VideoFileClip("../dataset/data/wind_gui/2_2/best.mp4") # or .avi, .webm, .gif ...

for idx, i in enumerate(clip.iter_frames()):
    #print(i.shape)
    if idx == 0 or idx == 20 or idx == 40 or idx == 59:
        cv2.imwrite("{}.png".format(idx), i[..., ::-1])
