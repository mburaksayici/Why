""" PyTorch CNN Layer utilities"""

import tensorflow as tf 
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_layers_type(model):
    """
    Returns model layer types dict of given Keras Model.
    """
    model_layers = {}
    idx = 0
    for name, layer in iter(model.named_modules()):
        model_layers[idx] = {"layer_name": name,"is_cnn":True if isinstance(layer, nn.Conv2d) else False }
        idx += 1
    return model_layers


def separate_model(model,layer_index=None):
    """
    Separates model into explaining, and post-explain models.
    """
    model_layers = get_layers_type(model)

    # Use last CNN if index not given.
    if not layer_index:
        layer_index = [i for i in list(reversed(list(model_layers.keys()))) if model_layers[i]["is_cnn"]==True][0]


    explaining_conv_layer_model = self.model.features[:layer_index]
    post_explain_model = self.model.features[layer_index:]

    return explaining_conv_layer_model, post_explain_model
























    explaining_conv_layer = model.get_layer(model_layers[layer_index]["layer_name"])
    explaining_conv_layer_model = tf.keras.Model(model.inputs, explaining_conv_layer.output)

    post_explain_input = tf.keras.Input(shape=explaining_conv_layer_model.output.shape[1:])

    x = post_explain_input
    for layer_name in model.layers[layer_index:]:
        x = model.get_layer(layer_name)(x)

    post_explain_model = tf.keras.Model(post_explain_input, x)

    return explaining_conv_layer_model, post_explain_model
