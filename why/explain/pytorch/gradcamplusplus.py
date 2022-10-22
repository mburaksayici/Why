""" GradCam Explainability"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..explain_utils import *
from .utils import PyTorchUtils, PyTorchGradCamModel


class GradCamPlusPlus:
    def __init__(self, model) -> None:

        self.model = model
        self.utils = PyTorchUtils()

    def _construct_gradcam_model(self, target_layer):
        model = PyTorchGradCamModel(self.model, target_layer)
        model.eval()
        return model

    def explain(
        self,
        input_array,
        explain_class=None,
        layer_index=None,
        heatmap_size=None,
        return_class=False,
    ):
        if layer_index is None:
            layer_index = self.utils.get_explainable_layers(self.model)[-2]

        gcmodel = self._construct_gradcam_model(layer_index)
        out, explaining_conv_layer_output = gcmodel(input_array)

        # If softmax not included, softmax the output. In PyTorch, softmax sometimes added to loss function.
        if out.max() > 1:
            out = torch.nn.functional.softmax(out)

        if explain_class is None:
            explain_class = out[0].argmax().item()

        loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([explain_class])))
        loss.backward()

        grads = gcmodel.get_act_grads().detach()

        score = out[0, explain_class]

        # Ref to original implementation : https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
        # Second Grads
        grads = grads * out[0, explain_class].exp()
        grads_2 = grads**2
        grads_3 = grads**3

        global_sum = explaining_conv_layer_output.sum((2, 3))

        alpha_num = grads_2
        alpha_denom = 2 * grads_2 + (global_sum.reshape((1, -1, 1, 1)) * grads_3)
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones(alpha_denom.shape)
        )
        alphas = alpha_num / alpha_denom

        weights = torch.clamp(grads, 0)

        # Convert to numpy to stick with original implementation
        weights = weights.detach().numpy()
        alphas = alphas.detach().numpy()

        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = alphas_thresholding.sum((0, 2, 3))
        alpha_normalization_constant_preprocessed = np.where(
            alpha_normalization_constant != 0.0,
            alpha_normalization_constant,
            np.ones(alpha_normalization_constant.shape),
        )

        alphas /= alpha_normalization_constant_preprocessed.reshape(1, -1, 1, 1)

        deep_linearization_weights = np.sum(
            (weights * alphas).reshape((-1, grads.shape[1])), axis=0
        )
        deep_linearization_weights = deep_linearization_weights.reshape(
            1, grads.shape[1], 1, 1
        )
        # Detach acts to numpy
        explaining_conv_layer_output = explaining_conv_layer_output.detach().numpy()

        heatmap = np.sum(
            deep_linearization_weights * explaining_conv_layer_output[0], axis=1
        )[0]

        # Passing through ReLU
        explanation = np.maximum(heatmap, 0)

        # Get image informations
        shape_list = list(input_array.shape)
        image_size = [i for i in shape_list if i > 4]
        if heatmap_size:
            image_size = heatmap_size
        channel = max([i for i in shape_list if i < 4])

        visualization = visualize(explanation, image_size, channel)

        if return_class:
            return visualization, explain_class
        return visualization
