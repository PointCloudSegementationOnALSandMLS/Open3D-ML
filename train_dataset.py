import open3d as o3d
import open3d.ml.torch as ml3d
import ml3d.datasets.essen as essen
import open3d.ml as _ml3d
import logging

cfg_file= '' #Load the config file  ./ml3d/configs/randlanet_essen_aerial.yml'

config = _ml3d.utils.Config.load_from_file(cfg_file)
# here we load the Essen dataset for an other datset, like Tronto3D we could use
# dataset = ml3d.datasets.Toronto3D(**cfg.dataset)
dataset = essen.Essen(**config.dataset)

framework= "torch"
Model = _ml3d.utils.get_module("model", config.model.name, framework)
model = Model(**config.model)

Pipeline = _ml3d.utils.get_module("pipeline", config.pipeline.name, framework)
pipeline = Pipeline(model, dataset, **config.pipeline)

#Run the training
pipeline.run_train()
