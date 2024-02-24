import open3d as o3d
import numpy as np
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import open3d.ml as _ml3d
import os
from os.path import exists, join, dirname 
import logging
import json

cfg_file= '' #Set the path to the config file './ml3d/configs/randlanet_toronto3d.yml'
ckpt_path_r = '' #Set the path to the "./randlanet_toronto.pth"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
cfg.model.ckpt_path = ckpt_path_r

model = ml3d.models.RandLANet(**cfg.model)
# To use KPConv use the following line instead line 15
#model = ml3d.models.KPFCNN(**cfg.model)

pipeline_r = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)
pipeline_r.load_ckpt(model.cfg.ckpt_path)
complete_labels= []
complete_predictions = []

array= np.load("") #load numpy array, like np.load("Essen/mobile/normalized/validation/gird_6.npy"). Here a train, validation or test grid can be loaded.
points= array[:,:3]
labels = array[:, 3]
feat = array[:, 4]
data = {
            'name': "essen",
            'point': points,
            'feat': feat,
            'label': labels,
        }   
results_r = pipeline_r.run_inference(data)
pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
# Fill "unlabeled" value because predictions have no 0 values.
pred_label_r[0] = 0


predictions_json = {
    'name': "essen",
    "label": labels.tolist(),
    "pred": pred_label_r.tolist(),
}
json.dump( predictions_json, open( "result.json", 'w' ) )