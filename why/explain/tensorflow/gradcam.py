""" GradCam Explainability"""

import tensorflow as tf
import numpy as np

from .utils import *
from ..explain_utils import *


class GradCam:
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
                loss = preds[:, 0]
                if explain_class:
                    explain_class_channel = preds[:, explain_class]
                else:
                    top_pred_index = tf.argmax(preds[0])
                    explain_class_channel = preds[:, top_pred_index]

            grads = tape.gradient(explain_class_channel, explaining_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        explaining_conv_layer_output_0 = explaining_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            explaining_conv_layer_output_0[:, :, i] *= pooled_grads[i]

        # Average over all the filters to get a single 2D array
        explanation = np.mean(explaining_conv_layer_output_0, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values

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
