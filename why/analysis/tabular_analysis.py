import os 
import logging

import pandas as pd 
from tqdm import tqdm
import numpy as np
from sklearn.metrics import  classification_report
from PIL import Image 

from why.explain.explain_utils import get_model_framework, generate_random_name
from why.explain import Explain

class TabularAnalysis:
    def __init__(self, model, preprocess_function) -> None:
        self.preprocess_function = preprocess_function
        self.model = model
        self.model_framework = get_model_framework(model)

    def _stack(self, input):
        if self.model_framework == "pytorch":
            import torch
            return torch.vstack(input)
        elif self.model_framework in ["tensorflow", "keras"]:
            # import pdb;pdb.set_trace()
            return np.vstack(np.array(input))

    def _predict(self, preprocessed_image):

        if self.model_framework == "pytorch":
            import torch
            return torch.nn.functional.softmax(self.model(preprocessed_image)).detach().numpy()

        elif self.model_framework in ["tensorflow", "keras"]:
            return self.model.predict(preprocessed_image)#.numpy()

    def _batch_prediction(self, training_data,batch_size):
        #import pdb;pdb.set_trace()

        prediction_results = []
        for i in tqdm(range(0, len(training_data), batch_size)):
            batch_list = []
            ground_truth_list = []
            image_path = []
            for data in training_data[i:i+batch_size]:
                # Prepare batch
                preprocessed_image = self.preprocess_function(data["image_path"])
                batch_list.append(preprocessed_image)
                ground_truth_list.append(data["class"])
                image_path.append(data["image_path"])

            preprocessed_batch = self._stack(batch_list)

            prediction = self._predict(preprocessed_batch)

                
            for ix in range(len(ground_truth_list)):
                prediction_results.append({"image_path":image_path[ix],"ground_truth":ground_truth_list[ix],"prediction":prediction[ix].argmax(),"logits":prediction[ix]
                  })

        predictions_df = pd.DataFrame(prediction_results)
        return predictions_df

    def _save_prediction(self, predictions_df, analysis_path):
        try:
            os.makedirs(analysis_path, exist_ok=True)
            predictions_df.to_csv(os.path.join(analysis_path, "results.csv"))
        except Exception as exc:
            random_name = generate_random_name(10)
            os.makedirs(analysis_path, exist_ok=True)
            predictions_df.to_csv(os.path.join(random_name, "results.csv"))
            logging.info(f"Couldn't save to the given path, saving results to : {random_name}, reason ::  {exc}")

    def _generic_performance_analysis(self, predictions_df):
        classification_report_ = classification_report(predictions_df["ground_truth"] , predictions_df["prediction"],output_dict=True)
        classification_report_df = pd.DataFrame(classification_report_).transpose()
        print(classification_report)
        return classification_report_df

    def _explain_prediction(self, image_path, method, explain_class):
        # Initialize explain object
        explanation_obj = Explain(self.model)

        # Get original image
        original_image = Image.open(image_path)

        # Get preprocessed image
        preprocessed_image = self.preprocess_function(image_path)
    

        heatmap = explanation_obj.explain(preprocessed_image,method=method, explain_class=explain_class)
        return explanation_obj.overlay_heatmap( original_image, heatmap, image_size=(224,224), alpha=0.5, colormap_name="jet", return_bytes=False)


    def _false_prediction_explaining_area_analysis(self, predictions_df, method="gradcam"):
        wrong_predictions_df = predictions_df[predictions_df["ground_truth"] != predictions_df["prediction"]][["ground_truth","prediction"]].sort_values("ground_truth")
        wrong_predictions_df = wrong_predictions_df.assign(freq=wrong_predictions_df.groupby('ground_truth')['ground_truth'].transform('count'))\
        .sort_values(by=['freq','ground_truth'],ascending=[False,True])

        confusion_report = """"""

        for idx, gt in enumerate(wrong_predictions_df["ground_truth"].unique()):
            inclass_confusions = wrong_predictions_df[wrong_predictions_df["ground_truth"] == gt]["prediction"].value_counts().to_dict()
            confusion_report += f"{idx+1}. most confused class is {gt}: \n"
            for key, value in inclass_confusions.items():
                confusion_report += f"\t It's mostly confused by class {key}: {value} times \n"
        print(confusion_report)

        return wrong_predictions_df

    def analyse(self,training_data, batch_size=16,analysis_path="analysis/"):
        predictions_df = self._batch_prediction(training_data,batch_size)
        self._save_prediction(predictions_df, analysis_path)
        classification_report_df = self._generic_performance_analysis(predictions_df)
        wrong_predictions_df = self._false_prediction_explaining_area_analysis(predictions_df)

        return predictions_df, classification_report_df, wrong_predictions_df
        