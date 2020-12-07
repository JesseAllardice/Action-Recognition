# Author: Jesse Allardice (12/2020)
"""
Person Module:
Holds all the information for a person which is extracted from a
webcam feed.
"""
# Standard packages
import os
import sys
import cv2
import numpy as np
import time
from collections import deque

# unique modules
from predictors import posepredictor # TODO: creat predictors Package
# TODO: create predictor ABC module # TODO: or inherient from scikit-learn estimator?
# TODO: creat PosePredictor module
from predictors import freqpredictor # TODO: Create FreqPredictor Module
from predictors import actionpredictor # TODO: Create ActionPredictor module

class Person():
    def __init__(self, ):
        # initialise the person variables
        self.image_deque = None
        self.pose_deque = None
        self.freq_deque = None
        self.action_deque = None
        # flags to indicate if each deque is up to date with the image_deque
        self.image_deque_up_to_date = False
        self.pose_deque_up_to_date = False
        self.freq_deque_up_to_date = False
        self.action_deque_up_to_date = False
        # predictors
        self.pose_predictor = None
        self.freq_predictor = None
        self.action_predictor = None

    def update_image_deque(self, user_image: np.ndarray, propogate=False):
        # add an image to the persons image deque
        # set the image deque up to date
        self.image_deque_up_to_date = True
        # set the pose, freq and action deques to not up to date
        self.pose_deque_up_to_date = False
        self.freq_deque_up_to_date = False
        self.action_deque_up_to_date = False

        if propogate:
            # culculate the pose deque
            # culculate the freq
            # culculate the action
            pass

    def set_image_deque(self, user_deque, propogate=False):
        self.image_deque = user_deque
        # set the image deque up to date
        self.image_deque_up_to_date = True
        # set the pose, freq and action deques to not up to date
        self.pose_deque_up_to_date = False
        self.freq_deque_up_to_date = False
        self.action_deque_up_to_date = False
        if propogate:
            # culculate the pose deque
            # culculate the freq
            # culculate the action
            pass

    def update_pose_deque(self, ):
        if not self.pose_deque_up_to_date:
            # predict pose from image_deque
            pass
        self.pose_deque_up_to_date = True
        # print("pose deque is up to date")

    def update_freq_deque(self, ):
        if not self.freq_deque_up_to_date:
            # predict freq from pose_deque
            pass
        self.pose_deque_up_to_date = True
        # print("freq deque is up to date")

    def update_action_deque(self, ):
        if not self.action_deque_up_to_date:
            # predict action from pose_deque
            pass
        self.action_deque_up_to_date = True
        # print("action deque is up to date")

    def predict_pose(self, image_deque) -> np.ndarray:
        pass

    def predict_freq(self, pose_deque) -> float:
        pass

    def predict_action(self, pose_deque) -> str:
        pass

def main():
    test_person = Person()
    # test the initialisation
    # run on some saved data
    print(test_person)
    pass

if __name__ == "__main__":
    main()