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
from predictors.posepredictor import PosePredictor # TODO: creat predictors Package
# TODO: create predictor ABC module # TODO: or inherient from scikit-learn estimator?
# TODO: creat PosePredictor module
from predictors.freqpredictor import FreqPredictor # TODO: Create FreqPredictor Module
from predictors.actionpredictor import ActionPredictor # TODO: Create ActionPredictor module

class Person():
    """
    Initialisation methods
    """
    def __init__(self, n_frames, n_padding): # j_frames (recalculate rate)
        # initialise the person variables
        self.n_frames = n_frames
        self.n_padding = n_padding
        # initialise the person deques
        self.image_deque = deque(maxlen=n_frames)
        self.pose_deque = deque(maxlen=n_frames)
        self.freq_deque = deque(maxlen=n_frames)
        self.action_deque = deque(maxlen=n_frames)
        # flags to indicate if each deque is up to date with the image_deque
        self.image_deque_up_to_date = False
        self.pose_deque_up_to_date = False
        self.freq_deque_up_to_date = False
        self.action_deque_up_to_date = False
        # predictors
        self.pose_predictor = PosePredictor(num_threads=8)
        self.freq_predictor = None
        self.action_predictor = None

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

    """
    Update deque methods
    """
    def update_image_deque(self, user_image: np.ndarray, propogate=False):
        # add an image to the persons image deque
        self.image_deque.append(user_image)
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

    """
    Inference on predictors
    """
    def predict_pose(self, image_deque) -> np.ndarray:
        pass

    def predict_freq(self, pose_deque) -> float:
        pass

    def predict_action(self, pose_deque) -> str:
        pass

    """
    Image maniplulation methods
    """
    def get_image(self) -> np.ndarray:
        return self.image_deque[-1]

    def get_pose(self) -> np.ndarray:
        return self.pose_deque[-1]

    def get_freq(self) -> np.ndarray:
        return self.pose_deque[-1]

    def get_action(self) -> np.ndarray:
        return self.action_deque[-1]

    def overlay_pose(self, user_image) -> np.ndarray:
        temp_image = user_image
        print("overlay_pose not implemented.")
        return temp_image

    def overlay_freq(self, user_image) -> np.ndarray:
        temp_image = user_image
        print("overlay_freq not implemented.")
        return temp_image

    def overlay_action(self, user_image) -> np.ndarray:
        temp_image = user_image
        print("overlay_action not implemented.")
        return temp_image

    def get_person_overlay(self) -> np.ndarray:
        temp_image = self.get_image()
        temp_image = self.overlay_pose(temp_image)
        temp_image = self.overlay_freq(temp_image)
        temp_image = self.overlay_action(temp_image)
        print(len(self.image_deque))
        return temp_image

def main():
    test_person = Person(32, 512)
    # test the initialisation
    # run on some saved data
    print(test_person)
    pass

if __name__ == "__main__":
    main()