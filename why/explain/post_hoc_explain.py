import importlib
import os

from .pytorch import VisualCorrespondence

from .explain_utils import *


class VisualCorrespondenceExplainer:
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

    def _import_method(self):
        explanation_module = importlib.import_module(
            f"why.explain.{self.model_framework}.visual_correspondence"
        )
        explain_class = getattr(explanation_module, "VisualCorrespondence")
        return explain_class

    def setup(
        self,
        preprocess_function=None,
        training_data=None,
        artifacts_filename="visual_correspondence",
    ):
        """
        preprocess_function : function that takes image url and preprocess.
        training_data : training data in the {"image_path":"class0image.png","class":0,}
        """
        visual_correspondence_class = self._import_method()
        self.visual_corresponder = visual_correspondence_class(self.model)
        self.visual_corresponder.setup(
            preprocess_function, training_data, artifacts_filename=artifacts_filename
        )

    def explain(self, input_array=None, image_path=None):
        return self.visual_corresponder.explain(
            input_array=input_array, image_path=image_path
        )

    def overlay_heatmap(
        self,
        image_path,
        heatmap,
        filename=None,
        image_size=None,
        alpha=0.5,
        colormap_name=None,
        return_bytes=False,
    ):
        original_image = Image.open(image_path)
        heatmap_resized = Image.fromarray(
            ((heatmap / heatmap.max()) * 255).astype(np.uint8)
        ).resize(original_image.size, 4)
        heatmap_resized = np.array(heatmap_resized)
        heatmap_resized = np.stack(
            [heatmap_resized, heatmap_resized, heatmap_resized], -1
        )

        filename = image_path.split("/")[-1] + "_similar.png"

        return overlay_heatmap_on_original_image(
            original_image,
            heatmap_resized,
            filename=filename,
            image_size=original_image.size,
            alpha=alpha,
            colormap_name=colormap_name,
            return_bytes=return_bytes,
        )
