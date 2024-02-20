import time
import collections
from collections import OrderedDict
import logging

import torch

from utils.meter import AverageMeter


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    model = model.cuda()
    outputs = model(inputs)
    # print(outputs.shape)
    outputs = outputs.data.cpu()
    if outputs.dim() == 4: outputs = outputs.mean(dim=[2,3])
    elif outputs.dim() == 3: outputs = outputs[:, 0]
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    logger = logging.getLogger('reid.train')

    end = time.time()
    with torch.no_grad():
        for i, info in enumerate(data_loader):
            imgs = info['images']
            pids = info['targets']
            fnames = info['img_path']
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Extract Features: [{}/{}]\t'
                      'batchTime {:.3f}s ({:.3f}s)\t'
                      'DataTime {:.3f}s ({:.3f}s)\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels