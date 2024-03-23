import os
from scipy.io import loadmat
from glob import glob

from . import DATASET_REGISTRY
from .bases import ImageDataset
import pdb
import random
import numpy as np

__all__ = ['Thermalworld',]


@DATASET_REGISTRY.register()
class Thermalworld(ImageDataset):
    dataset_dir = "thermalworld_rgb"
    dataset_name = "thermalworld"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        pid_list = os.listdir(train_path)
        for pid_dir in pid_list:
            pid = self.dataset_name + "_" + pid_dir
            img_list = glob(os.path.join(train_path, pid_dir,  "*.jpg"))
            for img_path in img_list:
                camid = self.dataset_name + "_cam0"
                data.append([img_path, pid, camid])
        return data
