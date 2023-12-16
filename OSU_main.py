import cv2
from vmbpy import *

with VmbSystem.get_instance () as vmb:
    cams = vmb.get_all_cameras ()
    print(cams[0].get_model())
    print(cams[1].get_model())

    with cams[1] as cam:
        frame = cam.get_frame()
        frame.convert_pixel_format(PixelFormat.Mono8)
        cv2.imwrite('frame.jpg', frame.as_opencv_image())

