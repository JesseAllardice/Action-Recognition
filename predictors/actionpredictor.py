"""
inherients from predictor
"""
# Standard packages
import tensorflow as tf
import numpy as np
import time
import pickle
import marshal
import types
from collections import deque
from sklearn.linear_model import LogisticRegression


# unique modules
from predictors.predictor import Predictor

class ActionPredictor(Predictor):
    """
    Instantisation methods
    """
    def __init__(self, N_samples, recording_fps):
        # number of samples to expect
        self.n_samples = int(N_samples)
        # recording/sampling rate
        self.recording_fps = int(recording_fps)
        # pose specification
        self.num_keypoints = None
        self.dimension = None
        # inputs
        self.input_data = None
        self.kinetic = None

        # preprocessing function
        self.preprocess = None
        # action prediction model
        self.model = None
        # prep and model file
        self.prep_and_model_file = "models\\actionnet_flat_linear_preprocess_and_model_10_fps.pickle"
        # load the prep and model
        self.load_preprocessing_and_model(self.prep_and_model_file)

        # outputs
        self.action = None

    """
    Inheriteted abstract methods
    """
    def predict(self, data: deque) -> np.ndarray:
        # set the num_keypoints and dimension
        self.set_pose_specification(data[0])
        # convert to a np.ndarray
        self.kinetic = np.array([data])
        # preprocess the data to the required form for the model
        X = self.preprocess(self.kinetic, image_size=np.array([480, 640])) # TODO: get image_size from person object
        self.input_data = X
        # predict using the model
        y_hat = self.model.predict(X)
        self.action = y_hat[0]
        return np.array([y_hat[0]])


    # def transform(self): pass

    # def fit(self): pass

    # def fit_transform(self): pass

    """
    Methods
    """
    def load_preprocessing_and_model(self, model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        function_code = marshal.loads(data["preprocess_function"])
        self.preprocess = types.FunctionType(function_code, globals(), 'preprocess')
        self.model = data["model"]

    def set_pose_specification(self, pose_example: np.ndarray):
        self.num_keypoints, self.dimension = pose_example.shape

    """
    Set Methods
    """
    def set_recording_fps(self, recording_fps):
        self.recording_fps = int(recording_fps)

    def set_N_samples(self, N_samples):
        self.n_samples = int(N_samples)

    """
    Get Methods
    """
    def get_action(self):
        return self.action
    
    def get_kinetic(self):
        return self.kinetic
    
    def get_input(self):
        return self.input_data

    """
    Static Methods
    """

def main():
    pass

if __name__ == "__main__":
    main()