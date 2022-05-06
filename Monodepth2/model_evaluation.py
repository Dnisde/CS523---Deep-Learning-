
from __future__ import absolute_import, division, print_function
%matplotlib inline

import os
import numpy as np
import pandas as pd
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

ground_model = "mono_640x192"
model_path = "./log/finetuned_mono/models/"

class model_evaluation:

    # Original Monodepth2 model name: model_name = "mono_640x192"
    # Your pretrained model name: self_model_name = "weights_19"
    # Your test image: image_path = "assets/test_image.jpg"

    def __init__(self, ground_truth_model_name, model_name, model_path, image_path, GT):
        self.GT_model = ground_truth_model_name
        self.model = model_name
        self.path = model_path
        self.image_path = image_path
        if GT == 1:
            download_model_if_doesnt_exist(ground_truth_model_name)
            self.encoderPath = os.path.join("models", ground_truth_model_name, "encoder.pth")
            self.depth_decoderPath = os.path.join("models", ground_truth_model_name, "depth.pth")
        else:
            self.encoderPath = os.path.join(model_path, model_name, "encoder.pth")
            self.depth_decoderPath = os.path.join(model_path, model_name, "depth.pth")

    def load_Pretrained_Model(self, model):
        # Initialize Encoder and Decoder Structure
        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        # LOADING PRETRAINED MODEL
        loaded_dict_enc = torch.load(self.encoderPath, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(self.depth_decoderPath, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        # Note!!!!!!!: Modify here: .eval() ??
        encoder.eval()
        depth_decoder.eval()

        return encoder, depth_decoder

    def load_preprocess(self):
        # Load test image from image dataset and doing pre-processing:
        image_path = self.image_path

        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        loaded_dict_enc = torch.load(self.encoderPath, map_location='cpu')
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        # Resize the image into certain dimension:
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        return input_image_pytorch, original_width, original_height

    def prediction(self):
        # Get Specific Trained Encoder and Decoder from the network, and visualize the performance:
        encoder, decoder = self.load_Pretrained_Model(self.encoderPath, self.depth_decoderPath)
        # Using the path to preprocess test image dataset:
        input_image, input_width, input_height = self.load_preprocess(self.encoderPath, self.depth_decoderPath)
        with torch.no_grad():
            features = encoder(input_image)
            outputs = decoder(features)
        disp = outputs[("disp", 0)]
        return disp, input_image, input_width, input_height

    def depth_visualize(self):
        # Call Prediction:
        disp, input_image, width, height = self.prediction()
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (height, width), mode="bilinear",
                                                       align_corners=False)
        # Saving color-mapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(input_image)
        plt.title("Input", fontsize=22)
        plt.axis('off');

        plt.subplot(212)
        plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
        plt.title("Disparity prediction", fontsize=22)
        plt.axis('off');


    def confusion_matrix(self, model_name):
        data = {'y_Actual': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
                }

        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        print(df)

if __name__ == "__main__":
    original_model = model_evaluation(ground_model, "weights_19", model_path, 1)
    original_model.depth_visualize()
    our_model =  model_evaluation(ground_model, "weights_19", model_path, 0)
    our_model.depth_visualize()