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
        out, acts = gcmodel(input_array)

        activations = acts.detach()

        if explain_class is None:
            explain_class = out[0].argmax().item()

        loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([explain_class])))
        loss.backward()

        grads = gcmodel.get_act_grads().detach()

        score = out[0,explain_class]

          # Ref to original implementation : https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
        # Second Grads
        
        grads_2 = grads**2#*out[0,explain_class].exp()
        grads_3 = grads**3#*out[0,explain_class].exp()
        #breakpoint()
        activation_constants = acts.sum((2,3))

        #breakpoint()
        num = grads_2
        denom = 2* grads_2 + (activation_constants.reshape((1,-1,1,1))*grads_3)
        denom = torch.where(denom != 0.0, denom, torch.ones(denom.shape))

        aik = num/denom
        
        # Eliminate zeros on grads 
        grads = torch.max(grads,torch.tensor([0.]))

        weights = torch.where(grads!=0, aik, 0.0) 
        # (grads*aik).sum((0,2,3)).reshape(1,-1,1,1)alphas_thresholding = 

        threshold_aik = torch.where(grads, aik>0, 0)

        aik_norm_constant = torch.sum(torch.sum(threshold_aik,0),0)
        

        heatmap_base =  torch.clamp(grads*weights,min=0) 

        heatmap_j = torch.mean(heatmap_base, dim=1).squeeze()
        heatmap_j_max = heatmap_j.max(axis=0)[0]
        heatmap_j /= heatmap_j_max
        explanation = heatmap_j.detach().numpy()
        explanation = explanation

        shape_list = list(input_array.shape)
        image_size = [i for i in shape_list if i > 4]
        channel = max([i for i in shape_list if i < 4])

        visualization = visualize(explanation, image_size, channel)
        if return_class:
            return visualization, explain_class
        return visualization

        """
        # Ref to original implementation : https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
        # Second Grads
        
        grads_2 = (grads**2)*out[0,explain_class].exp()
        grads_3 = (grads_2**3)*out[0,explain_class].exp()
        #breakpoint()
        activation_constants = acts.sum((2,3))

        aik = grads_2 / (2* grads_2 + (activation_constants.reshape((1,-1,1,1))*grads_3) + 0.00000001)

        grads = torch.where(grads<0,0,grads)
        
        weights = (grads*aik).sum((0,2,3)).reshape(1,-1,1,1)
        heatmap_base =  torch.clamp(grads*weights,min=0) 

        heatmap_j = torch.mean(heatmap_base, dim=1).squeeze()
        heatmap_j_max = heatmap_j.max(axis=0)[0]
        heatmap_j /= heatmap_j_max
        explanation = heatmap_j.detach().numpy()
        explanation = explanation

        shape_list = list(input_array.shape)
        image_size = [i for i in shape_list if i > 4]
        channel = max([i for i in shape_list if i < 4])

        visualization = visualize(explanation, image_size, channel)
        if return_class:
            return visualization, explain_class
        return visualization
        """