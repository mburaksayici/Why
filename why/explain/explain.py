import importlib
import os

from .pytorch import *
from .tensorflow import *

from .explain_utils import *


class Explain:
    def __init__(self, model):
        self.model = model
        self.model_framework = self._get_model_framework(model)

    def _get_model_framework(self, model):
        model_class_str = str(type(model))
        if "torch" in model_class_str:
            return "pytorch"
        elif "tensorflow" in model_class_str or "keras" in model_class_str:
            return "tensorflow"
        else:
            return None

    @staticmethod
    def get_pythonic_cwd():
        return os.getcwd().strip("/").replace("/", ".")

    def _import_method(self, method):
        explanation_module = importlib.import_module(
            f"why.explain.{self.model_framework}.{method.lower()}"
        )
        explain_class = getattr(explanation_module, method)
        return explain_class

    def explain(
        self,
        input_array,
        explain_class=None,
        method="GradCam",
        layer_index=None,
        heatmap_size=None,
        return_class=False,
    ):
        explanation_class = self._import_method(method)
        explanation_obj = explanation_class(self.model)

        return explanation_obj.explain(
            input_array=input_array,
            explain_class=explain_class,
            layer_index=layer_index,
            heatmap_size=heatmap_size,
            return_class=return_class,
        )

    def overlay_heatmap(
        self,
        original_image,
        heatmap,
        filename=None,
        image_size=None,
        alpha=0.5,
        colormap_name="RdYlBu",
        return_bytes=False,
    ):
        return overlay_heatmap_on_original_image(
            original_image,
            heatmap,
            filename=filename,
            image_size=image_size,
            alpha=alpha,
            colormap_name=colormap_name,
            return_bytes=return_bytes,
        )

    def annotate(
        self,
        input_array,
        image_size,
        threshold=0.7,
        explain_class=None,
        method="GradCam",
        layer_index=None,
    ):
        explanation_class = self._import_method(method)
        explanation_obj = explanation_class(self.model)

        heatmap, explain_class = explanation_obj.explain(
            input_array=input_array,
            explain_class=explain_class,
            layer_index=layer_index,
            heatmap_size=None,
            return_class=True,
        )

        resized_heatmap = resize_heatmap_wo_original_image(
            heatmap,
            filename=None,
            image_size=image_size,
            alpha=1,
            colormap_name="jet",
            return_bytes=False,
        )
        np_heatmap = np.array(resized_heatmap)
        mask = np_heatmap > int(threshold * 255)
        return create_polygon(mask[:, :, 0])

    def extract_area(
        self,
        input_array,
        original_image,
        threshold=0.7,
        explain_class=None,
        method="GradCam",
        layer_index=None,
    ):
        explanation_class = self._import_method(method)
        explanation_obj = explanation_class(self.model)

        heatmap, explain_class = explanation_obj.explain(
            input_array=input_array,
            explain_class=explain_class,
            layer_index=layer_index,
            heatmap_size=None,
            return_class=True,
        )

        resized_heatmap = resize_heatmap_wo_original_image(
            heatmap,
            filename=None,
            image_size=original_image.size,
            alpha=1,
            colormap_name="jet",
            return_bytes=False,
        )
        np_heatmap = np.array(resized_heatmap)
        mask = (np_heatmap > int(threshold * 255))[:, :, 0]

        np_original_image = np.array(original_image)
        extracted_area = []
        for i in range(np_original_image.shape[-1]):
            extracted_area.append(mask * np_original_image[:, :, i])
        extracted_area = np.stack(extracted_area, 2)
        return Image.fromarray(extracted_area)

    def resize_heatmap(
        self,
        heatmap,
        filename=None,
        image_size=None,
        alpha=0.5,
        colormap_name="RdYlBu",
        return_bytes=False,
    ):
        return resize_heatmap_wo_original_image(
            heatmap,
            filename=None,
            image_size=image_size,
            alpha=alpha,
            colormap_name=colormap_name,
            return_bytes=False,
        )
