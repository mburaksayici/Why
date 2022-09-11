""" GradCam Explainability"""

import torch
import torch.nn as nn
import numpy as np

from ..explain_utils import *
from .utils import PyTorchUtils, PyTorchGradCamModel


class GradCam:
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
            layer_index = self.utils.get_explainable_layers(self.model)[-3]

        gcmodel = self._construct_gradcam_model(layer_index)
        out, acts = gcmodel(input_array)

        acts = acts.detach()

        if explain_class is None:
            explain_class = out[0].argmax().item()

        loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([explain_class])))
        loss.backward()

        grads = gcmodel.get_act_grads().detach()
        pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

        for i in range(acts.shape[1]):
            acts[:, i, :, :] *= pooled_grads[i]

        heatmap_j = torch.mean(acts, dim=1).squeeze()
        heatmap_j_max = heatmap_j.max(axis=0)[0]
        heatmap_j /= heatmap_j_max
        explanation = heatmap_j
        explanation = explanation.numpy()

        shape_list = list(input_array.shape)
        image_size = [i for i in shape_list if i > 4]
        channel = max([i for i in shape_list if i < 4])

        visualization = visualize(explanation, image_size, channel)
        if return_class:
            return visualization, explain_class
        return visualization
