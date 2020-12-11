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
    def __init__(self, n_frames, n_padding, recording_fps): # j_frames (recalculate rate)
        # initialise the person variables
        self.n_frames = n_frames
        self.n_padding = n_padding
        self.recording_fps = recording_fps
        # initialise the person deques
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
        self.pose_predictor = PosePredictor(num_threads=8)
        self.freq_predictor = FreqPredictor(
            N_samples=n_frames,
            N_padding=n_padding,
            recording_fps=recording_fps,
            freq_range=(0,300),
        )
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
        if self.image_deque is None:
            # initialise the deque with this user_image as an example
            self.image_deque = self.init_deque(self.n_frames, user_image)
        else: # add the image to the persons image deque
            self.image_deque.append(user_image)
        # set the image deque up to date
        self.image_deque_up_to_date = True
        # set the pose, freq and action deques to not up to date
        self.pose_deque_up_to_date = False
        self.freq_deque_up_to_date = False
        self.action_deque_up_to_date = False

        if propogate:
            # culculate the freq
            # culculate the action
            pass

    def update_pose_deque(self, ):
        if not self.pose_deque_up_to_date:
            # predict pose from image_deque
            current_pose = self.predict_pose(self.image_deque)
            if self.pose_deque is None:
                # initialise the deque with this current_pose as an example
                self.pose_deque = self.init_deque(self.n_frames, current_pose)
            else: # add the pose to the deque
                self.pose_deque.append(current_pose)
            self.pose_deque_up_to_date = True
        else: print("pose deque is already up to date")

    def update_freq_deque(self, ):
        if not self.freq_deque_up_to_date:
            # predict freq from pose_deque
            current_freq = self.predict_freq(self.pose_deque)
            if self.freq_deque is None:
                # initialise the deque with this current_pose as an example
                self.freq_deque = self.init_deque(self.n_frames, current_freq)
            else: # add the pose to the deque
                self.freq_deque.append(current_freq)
            self.freq_deque_up_to_date = True
        else: print("freq deque is up to date")

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
        return self.pose_predictor.predict(image_deque)

    def predict_freq(self, pose_deque) -> float:
        return self.freq_predictor.predict(pose_deque)

    def predict_action(self, pose_deque) -> str:
        pass

    """
    Image maniplulation methods
    """
    def get_image(self) -> np.ndarray:
        return self.image_deque[-1]

    def get_model_image(self) -> np.ndarray:
        temp = self.pose_predictor.input_data
        temp = np.squeeze(temp)
        return temp

    def get_pose(self) -> np.ndarray:
        return self.pose_deque[-1]

    def get_model_pose(self) -> np.ndarray:
        return self.pose_predictor.model_positions

    def get_freq(self) -> np.ndarray:
        return self.freq_deque[-1]

    def get_action(self) -> np.ndarray:
        return self.action_deque[-1]

    def overlay_pose(self, user_image) -> np.ndarray:
        temp_image = user_image
        temp_pose = self.get_pose()
        for i in range(temp_pose.shape[0]):
            y, x = tuple(temp_pose[i, :])
            temp_image = self.overlay_dot(temp_image, x, y)
        return temp_image

    def overlay_model_pose(self, user_image) -> np.ndarray:
        temp_image = user_image
        temp_pose = self.get_model_pose()
        for i in range(temp_pose.shape[0]):
            y, x = tuple(temp_pose[i,:])
            temp_image = self.overlay_dot(temp_image, x, y)
        return temp_image

    def overlay_freq(self, user_image) -> np.ndarray:
        temp_image = user_image
        temp_freq = self.get_freq()
        temp_image = self.overlay_freq_text(temp_image, temp_freq)
        return temp_image

    def overlay_action(self, user_image) -> np.ndarray:
        temp_image = user_image
        print("overlay_action not implemented.")
        return temp_image

    def get_person_overlay(self, report_freq, report_action, use_model=False, mirror=False) -> np.ndarray:
        if use_model:
            get_image = self.get_model_image
            overlay_pose = self.overlay_model_pose
        else:
            get_image = self.get_image
            overlay_pose = self.overlay_pose
        temp_image = get_image()
        temp_image = overlay_pose(temp_image)
        if mirror:
                temp_image = self.x_reflected_image(temp_image)
                temp_image = np.array(temp_image)
        if report_freq:
            temp_image = self.overlay_freq(temp_image)
        if report_action:
            temp_image = self.overlay_action(temp_image)
        #print(len(self.image_deque))
        return temp_image

    """
    Plot seperately
    """
    def plot_freq_spectrum(self, user_subplot):
        # plot the kinetic
        x = np.arange(self.freq_predictor.n_samples)
        y = self.freq_predictor.get_kinetic()
        user_subplot[1][0].scatter(x,y)
        user_subplot[1][0].set_xlim(0,self.freq_predictor.n_samples)
        # plot the spectrum
        x = self.freq_predictor.get_freq(per_min=True)
        y = self.freq_predictor.get_spectrum()
        user_subplot[1][1].scatter(x,y)
        user_subplot[1][1].set_xlim(self.freq_predictor.freq_range)

    """
    Static Methods
    """
    @staticmethod
    def x_reflected_image(user_image: np.ndarray) -> np.ndarray:
        return user_image[:,::-1,:]

    @staticmethod
    def init_deque(maxlen, example) -> deque:
        init_list = [example for i in range(maxlen)]
        init_deque = deque(init_list, maxlen=maxlen)
        return init_deque

    @staticmethod
    def overlay_dot(img:np.ndarray, x, y) -> np.ndarray:
        coordinates = (int(x), int(y))
        # Write some Text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = coordinates
        fontScale              = 1
        fontColor              = (56,185,158)
        lineType               = 2
        cv2.putText(
            img,
            "o", # draws a "o" not a circle at the moment.
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType
        )
        return img

    @staticmethod
    def overlay_freq_text(img:np.ndarray, freq) -> np.ndarray:
        # Write some Text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 1
        fontColor              = (55,55,55)
        lineType               = 2
        cv2.putText(
            img,
            str(int(freq[0])), # draws a "o" not a circle at the moment.
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType
        )
        return img

def main():
    test_person = Person(32, 512)
    # test the initialisation
    # run on some saved data
    print(test_person)
    pass

if __name__ == "__main__":
    main()