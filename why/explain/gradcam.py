""" GradCam Explainability"""

import tensorflow as tf
import numpy as np

from utils.keras_cnn_utils import *


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
            preds = explaining_conv_layer_model(explaining_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        ## will be moved to keras utils
        grads = tape.gradient(top_class_channel, explaining_conv_layer_model)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        explaining_conv_layer_output = explaining_conv_layer_model.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            explaining_conv_layer_output[:, :, i] *= pooled_grads[i]

        explaining_conv_layer_output_0 = explaining_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            explaining_conv_layer_output_0[:, :, i] *= pooled_grads[i]

        # Average over all the filters to get a single 2D array
        explanation = np.mean(explaining_conv_layer_output_0, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values

        return explanation