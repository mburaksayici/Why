""" GradCam Explainability"""

import torch
import torch.nn as nn
import numpy as np

from .utils import *


class GradCam:
    def __init__(self, model) -> None:

        self.model = model
        self.utils = PyTorchUtils()

    def _construct_gradcam_model(self, target_layer):
        return PyTorchGradCamModel(self.model, target_layer)

    def explain(self, input_array, explain_class=None, layer_name=None):
        if layer_name is None:
            layer_name = self.utils.get_explainable_layers(self.model)[-2]

        gcmodel = self._construct_gradcam_model(layer_name)
        out, acts = gcmodel(input_array)

        acts = acts.detach()

        if explain_class is None:
            explain_class = out[0].argmax().item()

        loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([explain_class])))
        loss.backward()

        grads = gcmodel.get_act_grads().detach()
        pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

        for i in range(acts.shape[1]):
            acts[:, i, :, :] += pooled_grads[i]

        heatmap_j = torch.mean(acts, dim=1).squeeze()
        heatmap_j_max = heatmap_j.max(axis=0)[0]
        heatmap_j /= heatmap_j_max
        explanation = heatmap_j
        return explanation
