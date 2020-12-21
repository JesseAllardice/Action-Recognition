# Author: Jesse Allardice (12/2020)

"""
ActionRecogniser:
wraps a run loop which collects webcam frames and stores the extracted
pose information in a person object.
"""

# Standard packages
import os
import sys
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# unique modules
import person

class ActionRecogniser():
    """
    Initialisation methods
    """
    def __init__(self, report_action, report_freq, web_cam_num=0):
        self.report_action = report_action
        self.report_freq = report_freq
        self.web_cam_num = web_cam_num
        # Set up webcam variables etc
        self.init_webcam()
        # create person object
        self.person = self.creat_person()
        # current frame
        self.current_frame = None
        # show the freq spectrum
        self.show_spectrum = False
        self.spectrum_figure = None

    def init_webcam(self, ):
        # webcam feed
        self.WEBCAM_FEED = {
            "WEBCAM" : None,
            "IMG_SIZE" : None,
        }
        # testing short-circuits
        self.TESTING = [True,]
        # presets
        self.PRESETS = {
            "GRAY_SCALE" : False,
            "COLLECT_VIDEO" : True,
            "LIMIT_FRAME_RATE" : True,
            "LIMIT_METHOD" : 1,# 0 using the time difference to control the fps.
            # 1 using a fixed multipule of the camera frame rate. Effectively down sampling.
            "FPS_LIMIT" : 6,
            "HARD_CODE_CAMERA_FPS" : False, # if you want to input the camera fps
            "CAMERA_FPS" : 30, # sets the hardcoded camera fps if used.
            "GUESS_FPS_STANDARD" : True, # uses the measured fps to compare to a standard list of fps values.
            "N_TEST" : 30, # number of frames to use to estimate the camera fps.
            "STANDARD_FPS_VALUES" : [10, 15, 20, 24, 25, 30, 60, 120], # list of standard camera fps
            "RECORDING_FPS" : None,
        }

    """
    Setup Webcam methods
    """
    def set_webcam_feed(self,):
        # set the webcama feed
        self.WEBCAM_FEED["WEBCAM"] = cv2.VideoCapture(self.web_cam_num)
        # get camera fps
        fps = self.get_camera_fps(
            cv2VideoCapture=self.WEBCAM_FEED["WEBCAM"],
            N_test=self.PRESETS["N_TEST"],
            guess_fps_standard=self.PRESETS["GUESS_FPS_STANDARD"]
        )
        print('Cameras max fps is determined to be:', fps)
        # set camera recording fps
        self.PRESETS["RECORDING_FPS"] = fps # TODO: currently collect at max frame rate
        # set record method
        # TODO: currently collect at max frame rate
        # get image resolution
        self.WEBCAM_FEED["IMG_SIZE"] = self.get_camera_resolution(self.WEBCAM_FEED["WEBCAM"])

    def get_camera_resolution(self, cv2VideoCapture) -> tuple:
        _, frame = cv2VideoCapture.read()
        return np.shape(frame)[:2]

    def get_camera_fps(self, cv2VideoCapture, N_test: int, guess_fps_standard: bool) -> int:
        """finds the frame per second the camera can record.

        Args:
            cv2VideoCapture (cv2.VideoCapture): the webcam video capturer
            N_test (int): the number of frames to collect and average over.
            guess_fps_standard (bool): if True, the standard fps is determined by squared error.

        Returns:
            int: average fps of the camera.
        """
        fps = self.calc_camera_fps(cv2VideoCapture, N_test)
        if guess_fps_standard:
            # find the closest standard value fps
            possible_values = np.array(self.PRESETS["STANDARD_FPS_VALUES"])
            squared_error = (possible_values - fps)**2
            min_index = np.argmin(squared_error)
            fps = int(possible_values[min_index])
        return fps

    def calc_camera_fps(self, cv2VideoCapture, N_test: int) -> float:
        """Calculates the average frame rate for N frames collected

        Args:
            cv2VideoCapture (cv2.VideoCapture): the webcam video capturer
            N_test (int): the number of frames to collect and average over.

        Returns:
            float: average fps
        """
        start_time = time.time()
        for _ in range(N_test):
            _, _ = cv2VideoCapture.read()

            # # exit the capture loop?
            # if cv2.waitKey(1) & 0XFF == ord('q'): # waitKey(1) waits 1 us and keys a entered key.
            # # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            #     # exit the videocap loop if user enters 'q'
            #     cv2VideoCapture.release()
            #     cv2.destroyAllWindows()
            #     return
        end_time = time.time()
        elapsed_time = end_time - start_time # total time in sec for N_test frames collected
        period = elapsed_time / (N_test-1)  # average period for N_test-1 periods.
        fps = 1./period # average fps for N_test-1 periods.
        return fps

    """
    Run methods
    """
    def run(self, ):
        # initialise:
        # setup the webcam_feed
        self.set_webcam_feed()
        # person
        # run loop
        while True:
            # record the loop time
            start_time = time.time()
            # collect frame
            frame = self.collect_and_store_frame()
            # pass to person
            self.person.update_image_deque(frame)
            # pass pose to pose-predictor
            self.person.update_pose_deque()
            # convert pose deque to seq/matix
            # analysis/predict pose
            if self.report_freq:
                # pass pose-seq to freq-predictor
                # predict freq
                self.person.update_freq_deque()
            if self.report_action:
                # pass pose-seq to action-predictor
                # predict action
                self.person.update_action_deque()
            # plot the freq spectrum
            self.plot_spectrum()
            # plot image with pose, freq and action
            self.plot_person(just_image=False, mirror=True)
            # check for user input
            user_input = self.check_user_input()
            if user_input == 'show':
                self.show_spectrum = True
            if user_input == 'hide':
                self.show_spectrum = False
            elif user_input == 'break':
                break
            # report the loop duration
            print(time.time()-start_time)

    def collect_and_store_frame(self, ):
        # collect and store the next video frame
        # record webcam frame
        _, frame = self.WEBCAM_FEED["WEBCAM"].read()
        self.current_frame = frame
        return frame

    def check_user_input(self, ) -> str:
        # checks for specific user inputs
        # start
        # break
        if cv2.waitKey(1) & 0XFF == ord('s'):
            # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            self.WEBCAM_FEED["WEBCAM"].release()
            cv2.destroyAllWindows()
            return 'show'
        if cv2.waitKey(1) & 0XFF == ord('h'):
            # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            self.WEBCAM_FEED["WEBCAM"].release()
            cv2.destroyAllWindows()
            return 'hide'
        if cv2.waitKey(1) & 0XFF == ord('q'):
            # & 0XFF truncats the last 8 bits, which then get compared to 'q'.
            # exit the videocap loop if user enters 'q'
            self.WEBCAM_FEED["WEBCAM"].release()
            cv2.destroyAllWindows()
            return 'break'

    """
    Managing Person object methods
    """

    def creat_person(self, n_frames=32, n_padding=512, recording_fps=10) -> person:
        # fully initialise a person object
        temp_person = person.Person(
            n_frames=n_frames,
            n_padding=n_padding,
            recording_fps=recording_fps
        )
        # fill the person with frames? #TODO: add prefill?
        return temp_person

    def update_person(self, ):
        # updates the person with a new image
        print("update_person not implemented")

    def calculate_pose(self, ):
        # pass pose to pose-predictor
        # convert pose deque to seq/matix
        # analysis/predict pose
        print("calculate_pose not implmented")

    def calculate_freq(self, ):
        # pass pose-seq to freq-predictor
        # predict freq
        print("calculate_freq not implmented")

    def calculate_action(self, ):
        # pass pose-seq to action-predictor
        # predict action
        print("calculate_action not implmented")

    def plot_person(self, just_image=False, mirror=False, user_person=None):
        if just_image:
            # plot the image
            overlay_image = self.current_frame
            # if mirror:
            #     overlay_image = self.x_reflected_image(overlay_image)
            cv2.imshow('frame:', overlay_image)
        else:
            if user_person is None:
                user_person = self.person
            # plot the image, pose, freq and action
            overlay_image = user_person.get_person_overlay(
                report_freq=self.report_freq,
                report_action=self.report_action,
                use_model=False,
                mirror=mirror,
            )
            # if mirror:
            #     overlay_image = self.x_reflected_image(overlay_image)
            cv2.imshow('frame:', overlay_image)

    def plot_spectrum(self, ):
        if self.show_spectrum:
            if self.spectrum_figure is None:
                self.spectrum_figure = plt.subplots(2, 1)
                self.spectrum_figure[0].show()
                self.spectrum_figure[0].canvas.draw()
            for ax in self.spectrum_figure[1]:
                ax.clear()
            self.person.plot_freq_spectrum(self.spectrum_figure)
            self.spectrum_figure[0].canvas.draw()
        else:
            if self.spectrum_figure is None:
                return
            else:
                self.spectrum_figure[0].close() # close the spectrum
                self.spectrum_figure = None

    """
    Static Methods
    """
    @staticmethod
    def x_reflected_image(user_image: np.ndarray) -> np.ndarray:
        return user_image[:,::-1,:]

def main():
    action_recogniser = ActionRecogniser(
        report_action=True,
        report_freq=True,
        web_cam_num=0
    )
    action_recogniser.run()
    print("Done performing Action Recognition.")

if __name__ == "__main__":
    main()
