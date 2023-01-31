import os
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV3Small


# TO DO : Create base folder and place base classes at there.
class BaseFeatureExtractor(ABC):

    @abstractmethod
    def load(
        self,
    ):
        """
        feature extractor load function
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, input, size):
        """
        preprocess function for prediction, function is defined here as it should work for all image types, and the below function is generalised enough.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input):
        """
        predict function which inferes the input
        data based on the model loaded.
        Needs to be overriden in child class with
        custom implementation.
        """
        raise NotImplementedError("Predict function is not implemeneted.")


class MobileNetV3SmallFeatureExtractor(BaseFeatureExtractor):
    """a Onnx inference class.
    Args:
      model (Any): prediction model instance.
      input_name (str): Onnx model input layer name.
      label_name (str): Onnx model output layer name.
    """
    def _convert_imageurl_to_rgb_pil(self, inp):
        inp = Image.open(inp)
        inp = np.array(inp)

        channel = [i for i in np.shape(inp) if i <= 4]
        channel = 0 if len(channel) == 0 else max(channel)

        inp = inp.astype(np.uint8)
        if channel == 1:
            inp = Image.fromarray(inp[:, :, 0]).convert("RGB")
        elif channel == 0:
            inp = Image.fromarray(inp).convert("RGB")
        else:
            inp = Image.fromarray(inp).convert("RGB")
        return inp

    def preprocess(self, input, size, batch=False):
        """
        preprocess function for prediction, function is defined here as it should work for all image types, and the below function is generalised enough.
        """
        if batch:
            return np.asarray(
                [
                    np.asarray(self._convert_imageurl_to_rgb_pil(inp).resize((size, size)))
                    for inp in input
                ]
            )
        input = np.asarray(self._convert_imageurl_to_rgb_pil(input).resize((size, size)))
        return np.expand_dims(input, 0)

    def load(
        self,
    ):
        try:
            self.model = MobileNetV3Small(weights="imagenet", include_top=False)
        except Exception as exc:
            raise Exception(
                f"Feature Extractor MobileNetV3SmallFeatureExtractor can't be  loaded :: trace {exc}"
            )

    def predict(self, np_image):
        return self.model.predict(np_image, verbose=0)