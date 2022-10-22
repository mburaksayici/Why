""" GradCam Explainability"""

import tensorflow as tf
import numpy as np

from .utils import *
from ..explain_utils import *


class GradCamPlusPlus:
    """
    Gradcam++, referenced from : https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
    Lots of different implementations with significant differences from paper/other implementations. Sticking to original one.
    """

    def __init__(self, model) -> None:
        self.model = model

    def explain(
        self,
        input_array,
        explain_class=None,
        layer_index=None,
        heatmap_size=None,
        separate=False,
        return_class=False,
    ):
        if separate:
            explaining_conv_layer_model, post_explain_model = separate_model(self.model)

            ## will be moved to keras utils
            with tf.GradientTape() as tape:
                inputs = input_array[np.newaxis, ...]
                explaining_conv_layer_output = explaining_conv_layer_model(inputs)
                tape.watch(explaining_conv_layer_output)
                preds = post_explain_model(explaining_conv_layer_output)
                if not explain_class:
                    explain_class = tf.argmax(preds[0])
                explain_class_channel = preds[:, explain_class]

            ## will be moved to keras utils
            grads = tape.gradient(top_class_channel, explaining_conv_layer_output)

        else:
            multioutput_model = create_multioutput_model(
                self.model, layer_index=layer_index
            )

            with tf.GradientTape() as tape:
                input_array = tf.cast(input_array, tf.float32)
                tape.watch(input_array)
                explaining_conv_layer_output, preds = multioutput_model(input_array)
                if explain_class:
                    explain_class_channel = preds[:, explain_class]
                else:
                    top_pred_index = tf.argmax(preds[0])
                    explain_class_channel = preds[:, top_pred_index]

            grads = tape.gradient(explain_class_channel, explaining_conv_layer_output)

        # Grads
        grads = grads * tf.exp(explain_class_channel)
        grads_2 = grads**2
        grads_3 = grads_2 * grads

        explaining_conv_layer_output = explaining_conv_layer_output.numpy()

        global_sum = np.sum(
            explaining_conv_layer_output[0].reshape((-1, grads[0].shape[2])), axis=0
        )

        alpha_num = grads_2[0]
        alpha_denom = grads_2[0] * 2.0 + grads_3[0] * global_sum.reshape(
            (1, 1, grads[0].shape[2])
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        alphas = alpha_num / alpha_denom

        weights = np.maximum(grads[0], 0.0)

        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = np.sum(
            np.sum(alphas_thresholding, axis=0), axis=0
        )
        alpha_normalization_constant_processed = np.where(
            alpha_normalization_constant != 0.0,
            alpha_normalization_constant,
            np.ones(alpha_normalization_constant.shape),
        )

        alphas /= alpha_normalization_constant_processed.reshape(
            (1, 1, grads[0].shape[2])
        )
        alphas = alphas.numpy()

        deep_linearization_weights = np.sum(
            (weights * alphas).reshape((-1, grads[0].shape[2])), axis=0
        )
        # print deep_linearization_weights
        heatmap = np.sum(
            deep_linearization_weights * explaining_conv_layer_output[0], axis=2
        )

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
