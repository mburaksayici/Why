""" GradCam Explainability"""

import tensorflow as tf
import numpy as np

from ..utils.keras_cnn_utils import *
from ..utils.pytorch_cnn_utils import *


class KerasGradCam:
    def __init__(self, model) -> None:
        import tensorflow as tf

        self.model = model

    def explain(self, input_array, layer_index=None):

        explaining_conv_layer_model, post_explain_model = separate_model(self.model)

        ## will be moved to keras utils
        with tf.GradientTape() as tape:
            inputs = input_array[np.newaxis, ...]
            explaining_conv_layer_output = explaining_conv_layer_model(inputs)
            tape.watch(explaining_conv_layer_output)
            preds = post_explain_model(explaining_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        ## will be moved to keras utils
        grads = tape.gradient(top_class_channel, explaining_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        explaining_conv_layer_output_0 = explaining_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            explaining_conv_layer_output_0[:, :, i] *= pooled_grads[i]

        # Average over all the filters to get a single 2D array
        explanation = np.mean(explaining_conv_layer_output_0, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values

        return explanation


class PyTorchGradCam:
    def __init__(self, model) -> None:
        import torch
        import torch.nn as nn

        self.model = model
        self.utils = PyTorchUtils()

    def _construct_gradcam_model(self, target_layer):
        return PyTorchGradCamModel(self.model, target_layer)

    def explain(self, input_array, explain_class=None, layer_name=None):
        if layer_name is None:
            layer_name = self.utils.get_explainable_layers(self.model)[-1]

        gcmodel = self._construct_gradcam_model(layer_name)
        out, acts = gcmodel(input_array)

        acts = acts.detach()

        if explain_class is None:
            explain_class = 0

        loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([600])))
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
