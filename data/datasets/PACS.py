import glob
import logging
import os.path as osp
import re
import warnings
import os

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PACS(ImageDataset):
    dataset_dir = 'PACS'
    dataset_name = "PACS"

    def __init__(self, root='datasets', testname=None, mode='train'):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        if testname is None:
            testname = 'sketch'
        self.test_dir = osp.join(self.data_dir, testname)
        self.train_dir = []
        for trainname in os.listdir(self.data_dir):
            if trainname == testname:
                continue
            train_dir = osp.join(self.data_dir, trainname)
            self.train_dir.append(train_dir)

        logger = logging.getLogger('reid.train')
        logger.info("Loading PACS...")
        logger.info("testset: {}".format(testname))

        train = []
        for train_dir in self.train_dir:
            train.extend(self.process_dir(train_dir))
        test = self.process_dir(self.test_dir, is_train=False)

        self.train = train
        self.query = test
        self.gallery = []

        # if self.train != []:
        self.num_classes = self.parse_data(self.train)

        if mode == 'train':
            self.data = self.train
        elif mode == 'test':
            self.data = self.test
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | test]'.format(self.mode))


    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        classes = []
        for i, cls in enumerate(os.listdir(dir_path)):
            pth = []
            pth.extend(glob.glob(osp.join(dir_path, cls, '*.jpg')))
            pth.extend(glob.glob(osp.join(dir_path, cls, '*.png')))
            img_paths.extend(pth)
            classes += [cls] * len(pth)

        data = [(img_pth, classes[i], 0) for i, img_pth in enumerate(img_paths)]

        return data

    def show_train(self):
        logger = logging.getLogger('reid')
        num_train_classes = self.parse_data(self.train)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  -------------------------------')
        logger.info('  subset   | # class | # images')
        logger.info('  -------------------------------')
        logger.info('  train    | {:5d}   | {:8d}'.format(num_train_classes, len(self.train)))
        logger.info('  -------------------------------')

    def show_test(self):
        logger = logging.getLogger('reid')
        num_test_classes = self.parse_data(self.query)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  -------------------------------')
        logger.info('  subset   | # class | # images')
        logger.info('  -------------------------------')
        logger.info('  test    | {:5d}    | {:8d}'.format(num_test_classes, len(self.query)))
        logger.info('  -------------------------------')

    def parse_data(self, data):
        classes = set()
        if len(data[0]) > 2:
            for _, pid, _ in data:
                classes.add(pid)
        else:
            for _, pid in data:
                classes.add(pid)
        return len(classes)