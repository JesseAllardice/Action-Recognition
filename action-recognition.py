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
import numpy as np
import time

# unique modules
import person # TODO: creat person module

class ActionRecogniser():
    def __init__(self, report_action, report_freq, web_cam_num=0):
        self.web_cam_num = web_cam_num
        self.report_action = report_action
        self.report_freq = report_freq
        # Set up webcam variables etc
        # TODO: set up webcam
        # create person object
        self.person = self.creat_person()

    def run(self, ):
        # initialise:
        # run loop
        # person
        while True:
            # collect frame
            # check for user input
            # pass to person
            # pass pose to pose-predictor
            # convert pose deque to seq/matix
            # analysis/predict pose
            if self.report_freq:
                # pass pose-seq to freq-predictor
                # predict freq
                pass
            if self.report_freq:
                # pass pose-seq to action-predictor
                # predict action
                pass
            # plot image with pose, freq and action

    def creat_person(self, ) -> person:
        # fully initialise a person object
        temp_person = None
        return temp_person

    def collect_frame(self, ):
        # collect and store the next video frame
        pass

    def check_user_input(self, ):
        # checks for specific user inputs
        # start
        # break
        pass

    def update_person(self, ):
        # updates the person with a new image
        pass

    def calculate_pose(self, ):
        # pass pose to pose-predictor
        # convert pose deque to seq/matix
        # analysis/predict pose
        pass

    def calculate_freq(self, ):
        # pass pose-seq to freq-predictor
        # predict freq
        pass

    def calculate_action(self, ):
        # pass pose-seq to action-predictor
        # predict action
        pass

    def plot_person(self, user_person=None):
        if user_person is None:
            user_person = self.person
        # plot the image, pose, freq and action



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