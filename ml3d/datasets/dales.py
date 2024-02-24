import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging
import open3d as o3d

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Dales(BaseDataset):
    """Toronto3D dataset, used in visualizer, training, or test."""

    def __init__(self,
                 dataset_path,
                 name='Dales',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[6997560, 178021561, 120818120, 2583281, 748890, 799886, 1512927, 276924, 56908533],
                 ignored_label_inds=[0],
                 train_files=['5145_54460.ply', '5100_54495.ply', '5180_54435.ply', '5185_54485.ply', '5095_54455.ply', '5190_54400.ply', '5145_54480.ply', '5160_54330.ply', '5130_54355.ply', '5095_54440.ply', '5145_54340.ply', '5085_54320.ply', '5135_54495.ply', '5145_54405.ply', '5150_54340.ply', '5140_54445.ply', '5185_54390.ply', '5145_54470.ply', '5110_54495.ply', '5115_54480.ply', '5165_54390.ply', '5110_54320.ply', '5110_54475.ply', '5080_54435.ply', '5105_54460.ply', '5105_54405.ply', '5165_54395.ply', '5180_54485.ply', '5110_54460.ply'],
                 val_files=['5100_54440.ply', '5100_54490.ply', '5135_54430.ply', '5140_54390.ply', '5080_54470.ply', '5150_54325.ply', '5135_54435.ply', '5155_54335.ply', '5175_54395.ply', '5120_54445.ply', '5080_54400.ply'],
                 test_files=['5100_54440.ply', '5100_54490.ply', '5135_54430.ply', '5140_54390.ply', '5080_54470.ply', '5150_54325.ply', '5135_54435.ply', '5155_54335.ply', '5175_54395.ply', '5120_54445.ply', '5080_54400.ply'],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         train_files=train_files,
                         test_files=test_files,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()

        self.dataset_path = cfg.dataset_path
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_files = [
            join(self.cfg.dataset_path, f) for f in cfg.train_files
        ]
        self.val_files = [join(self.cfg.dataset_path, f) for f in cfg.val_files]
        self.test_files = [
            join(self.cfg.dataset_path, f) for f in cfg.test_files
        ]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'Unclassified',
            1: 'Ground',
            2: 'Vegetation',
            3: 'Cars',
            4: 'Trucks',
            5: 'Power lines',
            6: 'Poles',
            7: 'Fences',
            8: 'Buildings'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return DalesSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class DalesSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))
        self.UTM_OFFSET = [627285, 4841948, 0]

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        data = o3d.t.io.read_point_cloud(pc_path).point

        points = data["positions"].numpy()
        points = np.float32(points)

        #feat = data["colors"].numpy().astype(np.float32)

        labels = data['class'].numpy().astype(np.int32).reshape((-1,))

        data = {'point': points, 'feat': None, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}

        return attr


DATASET._register_module(Dales)
