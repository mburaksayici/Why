""" GradCam Explainability"""

import numpy as np

import utils.keras_cnn_utils as k_utils
import  utils.pytorch_cnn_utils as pt_utils


class KerasGradCam:
    def __init__(self, model) -> None:
        import tensorflow as tf 
        self.model = model

    def explain(self, input_array, layer_index=None):

        explaining_conv_layer_model, post_explain_model = k_utils.separate_model(self.model)

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

class PyTorchGradCam:
    def __init__(self, model) -> None:
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        
        self.model = model

    def explain(self, input_array, class_index,layer_index=None):
        
        one_hot = np.zeros((1000,1))
        one_hot[class_index] = 1
        one_hot = np.reshape(one_hot,(1,1000))
        explaining_conv_layer_model, post_explain_model = pt_utils.separate_model(self.model)

        pred = self.model(input_array) 
        pred.backward(gradient=torch.tensor(one_hot), retain_graph=True)

        # pull the gradients out of the model
        gradients = vgg.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = vgg.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)


