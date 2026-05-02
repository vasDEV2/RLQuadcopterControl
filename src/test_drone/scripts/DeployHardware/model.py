# import torch
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation as R 


class LoadONNX:

    def __init__(self, model_path="/home/vasudevan/Desktop/flightmare/flightrl/examples/EpisodeV.onnx"):
        

        # Load the ONNX model
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, obs):

        result = self.sess.run(None, {self.input_name: obs})
        result = np.clip(result, -1, 1)

        return result
    