import torch
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
        result = torch.tensor(result)
        result = torch.tanh(result)

        return result

class ModelLoader:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model object directly
        self.model = torch.jit.load(checkpoint_path)
        # Access normalizer
        print(self.model)
        norm = self.model.normalizer

        print("Normalizer state_dict:")
        for k, v in norm.state_dict().items():
            print(k, v.shape, v)

        # self.model.to(self.device)
        self.model.eval()

    def predict(self, x):
        # x = x.to(self.device)
        # with torch.no_grad():
        # x = self.model.normalizer(x)
        return self.model(x)
        
# Initialize
# loader = ModelLoader("policy.pt")

# # Call later from some other function
# def run_inference(data):
#     with torch.no_grad():
#         data = data.float().unsqueeze(0)
#         output = loader.predict(data)

#     return output


# euler = np.array([0, 0, 0])

# r = R.from_euler("zyx", euler)

# matrix = r.as_matrix()

# matrix = np.reshape(matrix, (9,))

# # data = torch.tensor([1,2,32,3,5,5,9,0,8,2,2,2])
# data = torch.tensor([-9.6633e-02,  1.0877e-02,  1.6271e-01,  1.3211e-02,  2.5767e-02,
#           8.8313e-04,  4.1126e-04, -4.1149e-05, -1.0000e+00,  1.4601e-02,
#          -3.9544e-02, -0.871e+00])

# data = torch.zeros(18)
# data[3:12] = matrix
# data = data.astype(np.float32)
# # ./data = torch.randn(12)
# print(data)
# print(data.shape)
# d = run_inference(data)

# # d[:, 0] = d[:, 0]*1.9*2.7

# print(d)
