""" Keras CNN Layer utilities"""

import tensorflow as tf
import cv2
import numpy as np


def get_layers_type(model):
    """
    Returns model layer types dict of given Keras Model. Mainly used to find CNN layers which are appropriate to explain.
    """
    model_layers = {}
    for idx, layer in enumerate(model.layers):
        model_layers[idx] = {
            "layer_name": layer.name,
            "is_cnn": True if isinstance(layer, tf.keras.layers.Conv2D) else False,
        }
    return model_layers


def _get_explaining_layer(model, layer_index):
    # Use last CNN if index not given.

    model_layers = get_layers_type(model)
    if not layer_index:
        layer_index = [
            i
            for i in list(reversed(list(model_layers.keys())))
            if model_layers[i]["is_cnn"] == True
        ][0]
    return layer_index, model_layers


def separate_model(model, layer_index=None):
    """
    Separates model into explaining, and post-explain models.
    """

    layer_index, model_layers = _get_explaining_layer(model, layer_index=layer_index)

    explaining_conv_layer = model.get_layer(model_layers[layer_index]["layer_name"])
    explaining_conv_layer_model = tf.keras.Model(
        model.inputs, explaining_conv_layer.output
    )

    post_explain_input = tf.keras.Input(
        shape=explaining_conv_layer_model.output.shape[1:]
    )

    x = post_explain_input
    for layer_idx in list(model_layers.keys())[layer_index:]:
        x = model.get_layer(model_layers[layer_idx]["layer_name"])(x)

    post_explain_model = tf.keras.Model(post_explain_input, x)

    return explaining_conv_layer_model, post_explain_model


def create_multioutput_model(model, layer_index=None):
    """
    Creates model that gives explaining layers output, and model's normal output as well.
    This method is better than separating model into two. Separation and creation of two model is way problematic.
    """
    layer_index, model_layers = _get_explaining_layer(model, layer_index=layer_index)

    multioutput_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(model_layers[layer_index]["layer_name"]).output, model.output],
    )
    return multioutput_model
