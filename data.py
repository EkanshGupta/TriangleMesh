import os
import sys
import pickle
import copy
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from meshcnn.models.layers.mesh import Mesh
from meshcnn.util.util import is_mesh_file, pad


class SHREC16(Dataset):
    def __init__(self, partition, device, opt: dict):
        '''
        opt structure:
            'ninput_edges': int, num edges to use for meshcnn (will pad if higher than actual)
            'num_points': int, num verts to use for dgcnn (has to be at most the actual num verts)
            'dataroot': str
        '''
        super(SHREC16, self).__init__()
        self.partition = partition
        self.device = device

        self.ninput_edges = opt['ninput_edges']
        self.num_points = opt['num_points']
        self.root = opt['dataroot']
        self.dir = os.path.join(self.root)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, partition)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)

        self.mean = 0
        self.std = 1
        self.get_mean_std() # init self.mean, self.std, self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=None, hold_history=False, export_folder=None)
        pointcloud = mesh.vs[:self.num_points].T
        meta = {'mesh': mesh, 'label': label, 'pointcloud': pointcloud}

        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size


    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.pkl')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 5 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)

        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, partition):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(partition)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes


def collate_fn(batch, device, is_train):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})

    input_edge_features = torch.from_numpy(meta['edge_features']).float()
    pointcloud = torch.from_numpy(meta['pointcloud']).float()
    label = torch.from_numpy(meta['label']).long()
    meta['edge_features'] = input_edge_features.to(device).requires_grad_(is_train)
    meta['pointcloud'] = pointcloud.to(device).requires_grad_(is_train)
    meta['label'] = label.to(device)
    # meta['mesh'] already contains the reqd list of meshes
    return meta


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, partition, opt: dict):
        '''
        opt structure:
            'gpu_ids': list of ints, or None (for cpu)
            'batch_size': int (default: 16)
            'max_dataset_size': int (default: inf)
            'shuffle': bool. Whether to shuffle or not
            'num_threads': int

            'dataset_opt': dict (i.e. nested options), with structure as mentioned in class SHREC16
        '''
        device = torch.device('cuda:{}'.format(opt['gpu_ids'][0])) if opt['gpu_ids'] else torch.device('cpu')
        self.dataset = SHREC16(partition, device, opt['dataset_opt'])

        self.batch_size = opt['batch_size']
        self.max_dataset_size = opt['max_dataset_size'] if opt['max_dataset_size'] else np.inf
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=opt['shuffle'],
            num_workers=opt['num_threads'],
            collate_fn=functools.partial(
                collate_fn,
                device=device, is_train=(partition=='train')
            )
        )

    def __len__(self):
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data