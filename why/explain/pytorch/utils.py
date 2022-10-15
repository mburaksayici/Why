""" Keras CNN Layer utilities"""
import torch
import torch.nn as nn


class PyTorchGradCamModel(nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.model = model

        self.layerhook.append(
            getattr(model, layer_name).register_forward_hook(self.forward_hook())
        )

        for p in self.model.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out


class PyTorchUtils:
    def __init__(
        self,
    ):
        pass

    def get_explainable_layers(self, model):
        """
        xai_layers = []

        for name, modules in model.named_modules():
            if isinstance(modules, torch.nn.Sequential):
                xai_layers.append(name)

        TO DO: Work on blocked models
        """
        return list(model.__dict__["_modules"].keys())
