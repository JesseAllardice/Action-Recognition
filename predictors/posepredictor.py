"""
inherients from predictor
"""
# Standard packages
import time
from collections import deque
import numpy as np
import tensorflow as tf
import cv2

# unique modules
from predictors.predictor import Predictor

class PosePredictor(Predictor):
    """
    Instantisation methods
    """
    def __init__(self, num_threads=8):
        if tf.__version__ < '2.3.1':
            raise Exception("Tensorflow 2.3.1 or greater is needed for multi-tread")
        self.model_path = 'models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        self.output_stride = 32
        self.num_threads = num_threads
        # load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path,
            num_threads=num_threads,
        )
        # allocate tensors
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        # NxHxWxC, H:1, W:2
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        # inputs
        self.input_data = None
        # outputs
        self.output_data = None
        self.offset_data = None
        # image specfics
        self.image = None
        self.image_height = None
        self.image_width = None
        # initialise coordinates
        self.model_positions = None
        self.image_positions = None

    """
    Inheriteted abstract methods
    """
    def predict(self, data: deque) -> np.ndarray:
        # extract the image data and the input data
        self.set_image_and_input(data)
        # invoke model
        self.invoke_pose_prediction(self.input_data)
        # calculate the model and image positions
        self.calculate_coordinates()
        return self.image_positions

    # def transform(self): pass

    # def fit(self): pass

    # def fit_transform(self): pass

    """
    Methods
    """
    def set_image_and_input(self, data: deque):
        # use only the most recent image
        image = data[-1]
        # set the image parameters, image, h and w
        self.image = image
        self.image_height, self.image_width, _ = image.shape
        # extract the model's image size
        model_img_size = (self.input_height, self.input_width)
        # downsample the image to the size for the model input
        image = cv2.resize(image, model_img_size, interpolation=cv2.INTER_LINEAR)
        # add N dim
        input_data = np.expand_dims(image, axis=0)
        # rescale to [-1,1)
        if self.floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        self.input_data = input_data

    def predict_on_random(self) -> np.ndarray:
        # set the image_shape
        self.image_height = self.input_details[0]['shape'][1]
        self.image_width = self.input_details[0]['shape'][2]
        # get the model's input shape
        input_shape = self.input_details[0]["shape"]
        # creat a matrix of the correct shape filled with random values
        self.image = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        # creat a input matrix
        self.input_data = self.image # no reshaping required here
        # invoke model
        self.invoke_pose_prediction(self.input_data)
        # calculate the model and image positions
        self.calculate_coordinates()
        return self.image_positions

    def invoke_pose_prediction(self, input_data):
        # Set the tensors
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        # run inference
        #start_time = time.time()
        self.interpreter.invoke()
        #stop_time = time.time()
        #print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        # the function 'get_tensor()' returns a copy of the tensor data.
        # whereas, use 'tensor()' in order to get a pointer to the tensor.
        self.output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        self.offset_data = self.interpreter.get_tensor(self.output_details[1]['index'])

    def calculate_coordinates(self):
        # remove the first dimension
        output_results = np.squeeze(self.output_data)
        offset_results = np.squeeze(self.offset_data)
        # set the stride value
        output_stride = self.output_stride
        # calculate the coordinates from the output and offset
        scores = self.sigmoid(output_results)
        num_keypoints = scores.shape[2]
        heatmap_positions = []
        offset_vectors = []
        confidences = []
        for ki in range(0, num_keypoints):
            x, y = np.unravel_index(np.argmax(scores[:, :, ki]), scores[:, :, ki].shape)
            confidences.append(scores[x, y, ki])
            offset_vector = (offset_results[x, y, ki], offset_results[x, y, num_keypoints + ki])
            heatmap_positions.append((x, y))
            offset_vectors.append(offset_vector)
        model_positions = np.add(np.array(heatmap_positions) * output_stride, offset_vectors)
        self.model_positions = model_positions
        self.image_positions = self.model_to_image_positions(model_positions)

    def model_to_image_positions(self, model_positions: np.ndarray) -> np.ndarray:
        scaling_x = self.image_height / self.input_height
        scaling_y = self.image_width / self.input_width
        scaling_matric = np.diag([scaling_x, scaling_y])
        image_positions = model_positions @ scaling_matric
        return image_positions

    """
    Get Methods
    """
    def get_model_positions(self):
        if self.model_positions is None:
            raise Exception("model_positions not calculated yet")
        return self.model_positions

    def get_image_positions(self):
        if self.image_positions is None:
            raise Exception("image_positions not calculated yet")
        return self.image_positions

    def get_input_data(self):
        if self.input_data is None:
            raise Exception("input_data not initialised yet")
        return self.input_data

    """
    Static Methods
    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def main():
    posepredictor = PosePredictor()
    image_positions = posepredictor.predict_on_random()
    print(image_positions)

if __name__ == "__main__":
    main()