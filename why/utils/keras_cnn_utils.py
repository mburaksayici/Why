""" Keras CNN Layer utilities"""

import tensorflow as tf



def get_layers_type(model):
    """
    Returns model layer types dict of given Keras Model.
    """
    model_layers = {}
    for idx, layer in enumerate(model.layers):
        model_layers[idx] = {"layer_name": layer.name,"is_cnn":True if isinstance(layer, tf.keras.layers.Conv2D) else False }
    return model_layers

def separate_model(model,layer_index):
    """
    Separates model into explaining, and post-explain models.
    """
    model_layers = get_layers_type(model)

    explaining_conv_layer = model.get_layer(layer_index)
    explaining_conv_layer_model = tf.keras.Model(model.inputs, explaining_conv_layer.output)

    post_explain_input = tf.keras.Input(shape=explaining_conv_layer_model.output.shape[1:])

    x = post_explain_input
    for layer_name in [v["layer_name"] for k,v in model_layers.items() if k > layer_index]:
        x = model.get_layer(layer_name)(x)
    post_explain_model = tf.keras.Model(post_explain_input, x)

    return explaining_conv_layer_model, post_explain_model
