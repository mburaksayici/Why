import os

import fastdup
import numpy as np
from tqdm import tqdm
import pandas as pd


class DatasetAnalysis:
    FASTDUP_METRICS = ['blur','size','mean','min','max','unique','stdv']
    FASTDUP_CSV_NAMES = {"similarity":"similarity.csv", "duplicates":"similarity.csv", "outliers":"outliers.csv","stats":"atrain_stats.csv"}

    def __init__(self, training_data_dict, results_folder="data_results") -> None:
        self.training_data_dict = training_data_dict
        self.classes = np.unique([i["class"] for i in self.training_data_dict])
        self.results_folder = results_folder

    def run_fastdup_analysis(self, ):
        return self._run_classwise_analysis()


    def _run_classwise_analysis(self):
        class_all_dict = dict()
        for class_name in tqdm(self.classes):
            class_results_dict = dict()
            class_folder = f"{class_name}_results"
            class_results_path = os.path.join(self.results_folder,class_folder)

            class_images = [i["image_path"] for i in self.training_data_dict if i["class"]==class_name]


            fastdup.run(input_dir=class_images, work_dir=class_results_path, nearest_neighbors_k=5, turi_param='ccthreshold=0.96')    #main running function.

            class_results_dict["similarity"] = fastdup.create_similarity_gallery(similarity_file=os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["similarity"]),save_path=class_results_path)     #create visualization of top_k similar images assuming data have labels which are in the folder name
            class_results_dict["duplicates"] = fastdup.create_duplicates_gallery(os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["similarity"]), save_path=class_results_path)     #create a visual gallery of found duplicates
            class_results_dict["outliers"] = fastdup.create_outliers_gallery(os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["outliers"]),   save_path=class_results_path)       #create a visual gallery of anomalies
            
            for metric in self.FASTDUP_METRICS:
                class_results_dict["metric"] = fastdup.create_stats_gallery(class_results_path,   save_path=class_results_path,metric=metric)       #create a visual gallery of anomalies
            class_all_dict[class_name] = class_results_dict        
        # In feature change it. 
        return True

    def accumulate_results(self):
        accumulated_results = dict()
        for class_name in tqdm(self.classes):
            accumulated_results[class_name] = dict()
            class_folder = f"{class_name}_results"
            class_results_path = os.path.join(self.results_folder,class_folder)

            accumulated_results[class_name]["similarity"] = pd.read_csv(os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["similarity"]))
            accumulated_results[class_name]["outliers"] = pd.read_csv(os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["outliers"]))
            accumulated_results[class_name]["stats"] = pd.read_csv(os.path.join(class_results_path,self.FASTDUP_CSV_NAMES["stats"]))

        self.accumulated_results = accumulated_results
        return accumulated_results

    def _craft_std(self, stats_df, min=True):
        """
        Most colourful and less colourful first 5 images. Stats on image standard deviation.
        min=True is less colourful image.
        """
        return stats_df.sort_values("stdv",ascending=min).head(5)["filename"].tolist()


    def get_metrics(self, class_name, metric="outliers"):
        if not self.accumulated_results:
            self.accumulate_results()


        if metric in self.FASTDUP_METRICS:
            return self.accumulated_results[class_name]["stats"].sort_values(metric,ascending=False)
        elif metric in ["stdmin", "stdmax"]:
            metric = "min" == metric[3:]
            return self._craft_std(self.accumulated_results[class_name]["stats"])
        else:
            return self.accumulated_results[class_name][metric].sort_values("distance",ascending=False)["from"].values.tolist()












"""
from ._feature_extractor import MobileNetV3SmallFeatureExtractor

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle





def cluster_image(self, validation_data):
    
    feature_vector_dict = dict()

    feature_extractor = MobileNetV3SmallFeatureExtractor()
    feature_extractor.load()

    for img in validation_data:

        image_url = img["image_url"]

        prep_input = self.feature_extractor.preprocess(image_url, size=224)
        feature_vectors = self.feature_extractor.predict(prep_input)
        feature_vector_dict[image_id] = feature_vectors.tolist()[0]

    feature_vector_dict.values

    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096)

    # get the unique labels (from the flower_labels.csv)
    df = pd.read_csv('flower_labels.csv')
    label = df['label'].tolist()
    unique_labels = list(set(label))

    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=100, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    # cluster feature vectors
    kmeans = KMeans(n_clusters=len(unique_labels),n_jobs=-1, random_state=22)
    kmeans.fit(x)
    
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)


"""