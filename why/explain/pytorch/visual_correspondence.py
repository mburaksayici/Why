import pickle
import logging
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from numpy import matlib as mb

from ..explain_utils import *
from .utils import PyTorchUtils, PyTorchGradCamModel


class RunningParams(object):
    """
    Official Paper Implementation Parameters
    https://github.com/anguyen8/visual-correspondence-XAI/blob/bcbe601352ff1bb641d540681b972ceef172657b/EMD-Corr/src/params.py
    """
    def __init__(self):
        self.VISUALIZATION = False
        self.UNIFORM = False
        self.MIX_DISTANCE = False
        # Calculate distance of two images based on 5 patches only (not entire image)
        self.CLOSEST_PATCH_COMPARISON = False#True
        self.IMAGENET_REAL = False
        self.INAT = False
        self.Deformable_ProtoPNet = False

        self.DEEP_NN_TEST = True
        self.KNN_RESULT_SAVE = True

        self.UNEQUAL_W = False

        self.layer4_fm_size = 7

        # Feature space to be used
        self.DIML_FEAT = True
        self.RANDOM_SHUFFLE = True
        self.AP_FEATURE = True
        self.DUPLICATE_THRESHOLD = 0.9

        self.N_test = 50000
        self.K_value = 300
        self.MajorityVotes_K = 20

        if self.VISUALIZATION is True:
            self.AP_FEATURE = True
            self.IMAGENET_REAL = True
            self.CLOSEST_PATCH_COMPARISON = True
            self.K_value = 300
            self.MajorityVotes_K = 20
RunningParams = RunningParams()

class VisualCorrespondence:
    def __init__(self, model):
        self.model = model
        self.utils = PyTorchUtils()
        self.artifacts_present = False

    # TO DO : Move to data utils
    def _load_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    # Paper calculation related functions
    def _construct_feature_extractor(self, target_layer):
        model = PyTorchGradCamModel(self.model, target_layer)
        model.eval()
        return model

    def _create_knn(self, x, y, artifacts_filename):
        # Paper suggests k = 20
        nn_ = KNeighborsClassifier(n_neighbors=20, algorithm="auto", metric="cosine")
        x = np.array(x)
        x = x.reshape( *x.shape[:-4],-1 )

        nn_.fit(x, y)

        # Calculate accuracy
        pred_y = nn_.predict(x)

        score = accuracy_score(y, pred_y)
        logging.info(f"Accuracy score of kNN is {score}")

        with open(f"{artifacts_filename}_knn.pkl", "wb") as fp:
            pickle.dump(nn_, fp)

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        for i in range(100):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(
                K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)
            ).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        if i == 99:
            print("Sinkhorn no solutions!")
            solution = False
        else:
            solution = True
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T, solution

    def compute_emd_sim(
     self, K_value, fb_center, fb, use_uniform, num_patch
    ):
        cc_q2g_maps = []
        cc_g2q_maps = []
        
        query = fb[0].cpu().detach().numpy()
        for i in range(K_value):
            gallery = fb[i].cpu().detach().numpy()
            heatmap1, heatmap2 = self.compute_spatial_similarity(query, gallery)
            # 7x7
            cc_g2q_maps.append(torch.from_numpy(heatmap1))
            cc_q2g_maps.append(torch.from_numpy(heatmap2))

        # 51x7x7
        cc_q2g_maps = torch.stack(cc_q2g_maps, dim=0)
        cc_g2q_maps = torch.stack(cc_g2q_maps, dim=0)

        # 51x49
        cc_q2g_weights = torch.flatten(cc_q2g_maps, start_dim=1)
        cc_g2q_weights = torch.flatten(cc_g2q_maps, start_dim=1)

        N = fb.size()[0]  # 51
        R = fb.size()[2] * fb.size()[3]  # 7*7=49

        # 51x2048x7x7 -> 51x2048x49
        fb = torch.flatten(fb, start_dim=2)
        fb = torch.nn.functional.normalize(fb, p=2, dim=1)

        fb_center = torch.flatten(fb_center, start_dim=2)
        fb_center = torch.nn.functional.normalize(fb_center, p=2, dim=1)

        # 51x49x49
        sim = torch.einsum("cm,ncs->nsm", fb[0], fb_center).contiguous().view(N, R, R)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)

        if use_uniform:
            u = torch.zeros(N, R).fill_(1.0 / R)
            v = torch.zeros(N, R).fill_(1.0 / R)
        else:
            if RunningParams.UNEQUAL_W is True:
                sum_g2q_att = cc_q2g_weights.sum(dim=1, keepdims=True) + 1e-5
                u = cc_q2g_weights / sum_g2q_att
                v = cc_g2q_weights / sum_g2q_att
            else:
                u = cc_q2g_weights
                u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
                v = cc_g2q_weights
                v = v / (v.sum(dim=1, keepdims=True) + 1e-5)

        u = u.to(sim.dtype)
        v = v.to(sim.dtype)
        T, solution = self.Sinkhorn(K, u, v)

        if RunningParams.CLOSEST_PATCH_COMPARISON:
            dists = []
            for i in range(K_value + 1):
                pair_opt_plan = torch.flatten(T[i], start_dim=0).to("cpu")
                pair_sim = torch.flatten(sim[i], start_dim=0).to("cpu")
                sorted_ts = torch.argsort(pair_opt_plan)
                # sorted_top = sorted_ts[-num_patch:]
                sorted_top = sorted_ts[-num_patch:]
                opt_point_top = np.array([pair_opt_plan[idx] for idx in sorted_top])
                sim_point_top = np.array([pair_sim[idx] for idx in sorted_top])
                dist = torch.as_tensor(np.sum(opt_point_top * sim_point_top))
                dists.append(dist)
            sim = torch.stack(dists, dim=0)
        else:
            sim = torch.sum(T * sim, dim=(1, 2))
        return sim, cc_q2g_maps, cc_g2q_maps, T


    def compute_spatial_similarity(self, conv1, conv2):
        """
        Takes in the last convolutional layer from two images, computes the pooled output
        feature, and then generates the spatial similarity map for both images.
        """
        conv1 = conv1.reshape(-1, conv1.shape[1] * conv1.shape[2]).T
        conv2 = conv2.reshape(-1, conv2.shape[1] * conv2.shape[2]).T

        # conv2 = conv2.reshape(-1, 7 * 7).T

        pool1 = np.mean(conv1, axis=0)
        pool2 = np.mean(conv2, axis=0)
        out_sz = (int(np.sqrt(conv1.shape[0])), int(np.sqrt(conv1.shape[0])))
        conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
        conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
        im_similarity = np.zeros((conv1_normed.shape[0], conv1_normed.shape[0]))

        for zz in range(conv1_normed.shape[0]):
            repPx = mb.repmat(conv1_normed[zz, :], conv1_normed.shape[0], 1)
            im_similarity[zz, :] = np.multiply(repPx, conv2_normed).sum(axis=1)
        similarity1 = np.reshape(np.sum(im_similarity, axis=1), out_sz)
        similarity2 = np.reshape(np.sum(im_similarity, axis=0), out_sz)
        return similarity1, similarity2


    # classmethod so that we do not need to instantiate
    def compute_emd_distance(self, K_value, fb_center, fb, use_uniform, num_patch):
        fb = torch.from_numpy(fb)
        fb_center = torch.from_numpy(fb_center)

        sim, q2g_att, g2q_att, opt_plan = self.compute_emd_sim(
            K_value, fb_center, fb, use_uniform, num_patch
        )
        if RunningParams.MIX_DISTANCE:
            cosine_sim = torch.einsum("chw,nchw->n", fb[0], fb_center)
            cosine_sim_max = cosine_sim.max()
            cosine_sim = cosine_sim / cosine_sim_max
            return sim + cosine_sim, q2g_att, g2q_att, opt_plan
        else:
            return sim, q2g_att, g2q_att, opt_plan

    def _create_emd(self, fb_center, distance_dict):
        fb_list = [fb_center] + [i["feature"] for i in distance_dict]
        fb = np.vstack(fb_list) 
        fb_center = np.vstack([fb_center]*len(fb_list))
        
        num_patch = 5 # distance_dict[0]["shape"][2:]
        K_value = min(RunningParams.K_value, len(fb_list) )

        emd_distance, q2g_att, g2q_att, opt_plan = self.compute_emd_distance(
                        K_value  , fb_center, fb, RunningParams.UNIFORM, num_patch
                        )


        emd_distance = emd_distance[1:] # Remove the Query itself from distance array
        q2g_att = q2g_att[1:] # Remove the Query itself from heatmap array
        g2q_att = g2q_att[1:] # Remove the Query itself from heatmap array
        return emd_distance, q2g_att, g2q_att  



    def setup(self, preprocess_function, training_data, artifacts_filename="visual_correspondence"):
        features_exists = os.path.isfile(artifacts_filename+"_dist.pkl")
        knn_exists = os.path.isfile(artifacts_filename+"_knn.pkl")
        if (knn_exists and features_exists):
            self.preprocess_function = preprocess_function
            self.artifacts_present = True
            self.artifacts_filename = artifacts_filename
            
            layer_index = self.utils.get_explainable_layers(self.model)[-2]
            self.feature_extractor = self._construct_feature_extractor(layer_index)

        else:
            # If first time, kNN and features for training set will be extracted
            self.artifacts_filename
            self.preprocess_function = preprocess_function
            #  Create feature extractor model
            layer_index = self.utils.get_explainable_layers(self.model)[-2]
            self.feature_extractor = self._construct_feature_extractor(layer_index)
        
            distance_list = training_data.copy()
            logging.info("Creating feature extractions for Visual Correspondence")
            for image_dict in tqdm(distance_list):
                preprocessed_image = self.preprocess_function(image_dict["image_path"])
                model_prediction, feature = self.feature_extractor(preprocessed_image)
                    
                image_dict["feature"] = feature.detach().numpy()
                image_dict["shape"] =  np.array(feature.shape)
            
            with open(f"{artifacts_filename}_dist.pkl", "wb") as fp:
                pickle.dump(distance_list, fp)

            knn_x = [i["feature"] for i in distance_list]
            knn_y = [i["class"] for i in distance_list]
        
            self._create_knn(knn_x, knn_y, artifacts_filename)
        
        
    def explain(self, input_array=None, image_path=None):
        nn_ = self._load_pickle(f"{self.artifacts_filename}_knn.pkl")
        dist_ = self._load_pickle(f"{self.artifacts_filename}_dist.pkl")
        if image_path:
            input_array = self.preprocess_function(image_path)

        model_prediction, feature = self.feature_extractor(input_array)
        feature = feature.detach().numpy()
        # kNN already predicts the closest images. So we need to filter predicted class results first.
        # In the first stage, we select the N images having the lowest cosine distance
        closest_indices = nn_.kneighbors(feature.reshape(1, -1),  n_neighbors = 50)[-1].flatten().tolist()
        
        # Then, we'll rerank this 50 images, to 20 by emd.
    
        emd_dist_ = []
        ref_dict = dict()
        for  i, closest_index in enumerate(closest_indices):
            emd_dist_.append(dist_[closest_index])
            ref_dict[i] = closest_index

        emd_distance, q2g_att, g2q_att   = self._create_emd(feature, emd_dist_)
        
        for i in range(len(emd_dist_)):
            emd_dist_[i].pop("feature")
            emd_dist_[i].pop("shape")
            emd_dist_[i]["similarity_rank"] = i+1
            emd_dist_[i]["heatmap"] = q2g_att[i].detach().numpy()

        return emd_dist_

