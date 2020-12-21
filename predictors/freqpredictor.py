"""
inherients from predictor
"""
# Standard packages
import numpy as np
from collections import deque

# unique modules
from predictors.predictor import Predictor

class FreqPredictor(Predictor):
    """
    Instantisation methods
    """
    def __init__(self, N_samples, N_padding, recording_fps, freq_range=(0,175)):
        # number of samples to expect
        self.n_samples = int(N_samples)
        # length to pad fft to
        self.n_padding = int(N_padding)
        # recording/sampling rate
        self.recording_fps = int(recording_fps)
        # allowed freq range  (per min)
        self.freq_range = freq_range
        # pose specification
        self.num_keypoints = None
        self.dimension = None
        # inputs
        self.input_data = None
        self.kinetic = None

        # outputs
        self.per_pixel_spectrum = None
        self.spectrum = None
        self.freq = None
        self.freq_per_min = None
        self.argmax_spectrum_index = None
        self.argmax_spectrum_freq = None
        self.argmax_spectrum_freq_per_min = None

    """
    Inheriteted abstract methods
    """
    def predict(self, data: deque) -> np.ndarray:
        # set the num_keypoints and dimension
        self.set_pose_specification(data[0])
        # convert the deque to a ndarray
        video = self.deque_to_matrix(data, internal_coord=True)
        self.input_data = video
        # remove the mean/DC component
        video = self.filter_and_pad(video)
        # calculate the FFT components
        per_pixel_fft = self.pixel_wise_fft(video)
        # extract the fourier compoent amplitudes
        abs_fft = np.abs(per_pixel_fft)
        # take the mean spectrum in both keypoint, space and colour
        mean_abs_fft = np.mean(abs_fft, axis=1)
        mean_input_data = np.mean(video, axis=1)
        self.spectrum = self.colapse_color_channels(mean_abs_fft)
        self.kinetic = self.colapse_color_channels(mean_input_data)
        # cacluate the FFT freq
        self.freq = np.fft.fftfreq(n=self.n_padding) * self.recording_fps
        self.freq_per_min = self.freq * 60
        # find the argmax_index, freq and freq_per_min
        self.find_spectrum_peak_freq(self.spectrum, self.freq)
        # return the argmax_freq (per min)
        return self.argmax_spectrum_freq_per_min

    def filter_and_pad(self, data: np.ndarray):
        data = data - np.mean(data, axis=0)
        pad_len = self.n_padding - self.n_samples
        #data = np.pad(data, (pad_len, 0), 'edge')
        return data

    # def transform(self): pass

    # def fit(self): pass

    # def fit_transform(self): pass

    """
    Methods
    """
    def set_pose_specification(self, pose_example: np.ndarray):
        self.num_keypoints, self.dimension = pose_example.shape

    def find_spectrum_peak_freq(self, spectrum, freq):
        freq_range = self.freq_range
        freq_min_index = np.abs(freq * 60 - freq_range[0]).argmin()
        freq_max_index = np.abs(freq * 60 - freq_range[1]).argmin()
        restricted_freq = freq[freq_min_index:freq_max_index]
        restricted_spectrum = spectrum[freq_min_index:freq_max_index]
        peak_index = restricted_spectrum.argmax(axis=0)
        self.argmax_spectrum_index = peak_index
        peak_freq = restricted_freq[peak_index]
        self.argmax_spectrum_freq = peak_freq
        self.argmax_spectrum_freq_per_min = np.array([peak_freq * 60])

    def deque_to_matrix(self, user_deque: deque, internal_coord=True) -> np.ndarray:
        img_size = user_deque[0].shape
        if internal_coord:
            num_inter_coordinates = int((img_size[0]**2 - img_size[0])/2)
            video_size = [len(user_deque), num_inter_coordinates, img_size[1]] # len(files)
            video_matrix = np.zeros(video_size)
            for i in range(len(user_deque)):  # len(files)
                external_coord = user_deque[i]
                diffences = self.diff_matrix(external_coord)
                internal_coord = diffences[np.triu_indices(external_coord.shape[0], k=1)]
                video_matrix[i,:,:] = internal_coord
            return video_matrix
        else:
            video_size = [len(user_deque), *img_size] # len(files)
            video_matrix = np.zeros(video_size)
            for i in range(len(user_deque)):  # len(files)
                video_matrix[i,:,:] = user_deque[i]
            return video_matrix

    def pixel_wise_fft(self, user_ndarray, real_valued=False) -> np.ndarray:
        if real_valued:
            return np.fft.rfft(user_ndarray, n=self.n_padding, axis=0)
        else:
            return np.fft.fft(user_ndarray, n=self.n_padding, axis=0)

    """
    Set Methods
    """
    def set_recording_fps(self, recording_fps):
        self.recording_fps = int(recording_fps)

    def set_N_samples(self, N_samples):
        self.n_samples = int(N_samples)

    def set_N_padding(self, N_padding):
        self.n_padding = int(N_padding)

    def set_freq_range(self, freq_range):
        self.freq_range = freq_range

    """
    Get Methods
    """
    def get_spectrum(self):
        return self.spectrum

    def get_freq(self, per_min=True):
        if per_min:
            return self.freq_per_min
        else:
            return self.freq

    def get_kinetic(self):
        return self.kinetic

    """
    Static Methods
    """
    @staticmethod
    def colapse_color_channels(user_ndarry):
        if len(user_ndarry.shape) == 1:
            return user_ndarry
        else:
            return np.mean(user_ndarry, axis=-1)

    @staticmethod
    def diff_matrix(user_matrix):
        n, d = user_matrix.shape
        ones = np.ones((user_matrix.shape[0],1))
        v = user_matrix.reshape(n, 1, d)
        rv = np.einsum('ik..., kj -> ik...', v, ones) - np.einsum('kj..., ki -> jk...', v, ones)
        return rv

def main():
    freqpredictor = FreqPredictor(32, 512, 20)
    # freq = freqpredictor.predict_on_random() # not implemented yet
    print(freq)

if __name__ == "__main__":
    main()