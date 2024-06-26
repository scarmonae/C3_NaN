import cv2
import pdb
import keyboard

import os
from datetime import datetime

from vmbpy import *
os.environ["DISPLAY"] = ":0"

import copy
import queue
import threading
from typing import Optional

import cv2
import numpy

from vmbpy import *

FRAME_QUEUE_SIZE = 10   
FRAME_HEIGHT = 1944
FRAME_WIDTH = 2592

VIDEO_OUTPUT_DIR = 'videos'
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

def print_preamble():
    print('///////////////////////////////////////')
    print('/// C3_NaN project Kardashan Bee Routine ///')
    print('///////////////////////////////////////\n')
    print(flush=True)

def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame

def resize_if_required(frame: Frame) -> numpy.ndarray:
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame

def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame

def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass

def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)
    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = max_
        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        feat.set(val)

        msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
               'Using nearest valid value \'{}\'. Note that, this causes resizing '
               'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))

class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        try:
            self.cam.ExposureAuto.set('Once')
        except (AttributeError, VmbFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(self.cam.get_id()))

        try:
            self.cam.set_pixel_format(PixelFormat.Bgr8)
        except:
            self.cam.set_pixel_format(PixelFormat.Mono8)
            self.log.info('Pixel format incompatible with one camera. Pixel format changed to "Mono8"')

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

        try:
            with self.cam:
                self.setup_camera()

                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()

        except VmbCameraError:
            pass
        finally:
            try_put_frame(self.frame_queue, self.cam, None)
        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))

def scale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return scaled_image

class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, image_directory='captured_images'):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.current_frame = None
        self.image_directory = image_directory
        self.video_writer = None
        self.recording = False
        self.start_time = None
        self.last_frame_time = None

        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)

    def on_key_press(self, key):
        if key == ord('s'):
            if self.current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = os.path.join(self.image_directory, f'captured_image_{timestamp}.jpg')
                cv2.imwrite(file_name, self.current_frame)
                print(f'Image saved as {file_name}')
        elif key == ord('v'):
            if self.recording:
                self.stop_video_recording()
                self.recording = False
                print("Stopped video recording.")
            else:
                self.start_video_recording((self.scaled_frame.shape[1], self.scaled_frame.shape[0]))
                self.recording = True
                self.start_time = cv2.getTickCount()
                self.last_frame_time = self.start_time
                print("Started video recording.")

    def start_video_recording(self, frame_size):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = os.path.join(VIDEO_OUTPUT_DIR, f'video_{timestamp}.avi')
        self.video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), 20.0, frame_size)

    def stop_video_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit, <v> to start/stop recording'
        KEY_CODE_ENTER = 13

        frames = {}
        alive = True

        self.log.info('Thread \'FrameConsumer\' started.')

        while alive:
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)
                frames_left -= 1

            if frames:
                resized_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                adjusted_images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.shape[2] == 1 else img for img in resized_images]
                self.current_frame = numpy.concatenate(adjusted_images, axis=1)
                self.scaled_frame = scale_image(self.current_frame, 25)
                cv2.imshow(IMAGE_CAPTION, self.scaled_frame)

                if self.recording and self.video_writer:
                    current_time = cv2.getTickCount()
                    time_elapsed = (current_time - self.last_frame_time) / cv2.getTickFrequency()
                    if time_elapsed >= 1.0 / 20.0:
                        self.video_writer.write(self.scaled_frame)
                        self.last_frame_time = current_time

            else:
                self.current_frame = create_dummy_frame()
                cv2.imshow(IMAGE_CAPTION, self.current_frame)

            key = cv2.waitKey(10)
            if key == KEY_CODE_ENTER:
                cv2.destroyAllWindows()
                alive = False
            else:
                self.on_key_press(key)

        if self.recording:
            self.stop_video_recording()
        self.log.info('Thread \'FrameConsumer\' terminated.')

class MainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        consumer = FrameConsumer(self.frame_queue)

        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('Thread \'MainThread\' started.')

        with vmb:
            for cam in vmb.get_all_cameras():
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            vmb.register_camera_change_handler(self)
            consumer.start()
            consumer.join()
            vmb.unregister_camera_change_handler(self)

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()

        log.info('Thread \'MainThread\' terminated.')

if __name__ == '__main__':
    print_preamble()
    main = MainThread()
    main.start()
    main.join()
