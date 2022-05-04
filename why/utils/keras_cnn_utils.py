""" Keras CNN Layer utilities"""

import tensorflow as tf


def get_layers_type(model):
    """
    Returns model layer types dict of given Keras Model.
    """
    model_layers = {}
    for idx, layer in enumerate(model.layers):
        model_layers[idx] = {
            "layer_name": layer.name,
            "is_cnn": True if isinstance(layer, tf.keras.layers.Conv2D) else False,
        }
    return model_layers


def separate_model(model, layer_index=None):
    """
    Separates model into explaining, and post-explain models.
    """
    model_layers = get_layers_type(model)

    # Use last CNN if index not given.
    if not layer_index:
        layer_index = [
            i
            for i in list(reversed(list(model_layers.keys())))
            if model_layers[i]["is_cnn"] == True
        ][0]

    explaining_conv_layer = model.get_layer(model_layers[layer_index]["layer_name"])
    explaining_conv_layer_model = tf.keras.Model(
        model.inputs, explaining_conv_layer.output
    )

    post_explain_input = tf.keras.Input(
        shape=explaining_conv_layer_model.output.shape[1:]
    )

    x = post_explain_input
    for layer_idx in list(model_layers.keys())[layer_index:]:
        if model_layers[layer_idx]["layer_name"] in ["avg_pool", "predictions"]:
            print(model_layers[layer_idx]["layer_name"])
            x = model.get_layer(model_layers[layer_idx]["layer_name"])(x)

    post_explain_model = tf.keras.Model(post_explain_input, x)

    return explaining_conv_layer_model, post_explain_model
