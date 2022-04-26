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

from meshcnn.util.util import print_network
from meshcnn.models import networks
from model import CombinedMeshClassifier

class ClassifierWrapper:
    def __init__(self, is_train, opt: dict):
        '''
        opt structure:
            'gpu_ids': list of ints, or None (for cpu)
            'checkpoints_dir': str, parent folder for storing models (expt name will be a subfolder here)
            'expt_name': str, name of expt; decides where to store models etc
            'continue_train': bool, whether to resume training from a given epoch (if training).
            'which_epoch': str, which epoch to load if testing or resuming training (default: 'latest'; can enter a number)

            'network_opt': dict (nested options) for model architecture. see CombinedMeshClassifier for details

            'net_init_opt': dict (nested options) for weight init, structure:
                'init_type': str, type of initalization to use, among [normal|xavier|kaiming|orthogonal]. 'normal' is default
                'init_gain': float, gain used for normal/xavier/orthogonal inits. 0.02 is default

            'lr_schedule_opt': dict (nexted options) for LR schedule, structure:
                'lr': float, init LR, default 0.0002
                'lr_policy': str, type of schedule, among lambda|step|plateau, default 'lambda'
                'lr_decay_iters': int, decay by gamma every lr_decay_iters iterations (if step policy). default 50
                'lr_decay_gamma': float, the gamma for decay for step policy. default 0.1
                'epoch_count': int, the starting epoch count (if lambda policy). default 1
                'niter': int, num epochs to be at starting LR (if lambda policy). default 100
                'niter_decay': int, num epochs to decay LR linearly to zero (if lambda policy). default 500.
        '''
        self.gpu_ids = opt['gpu_ids']
        self.is_train = is_train

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt['checkpoints_dir'], opt['expt_name'])
        os.makedirs(self.save_dir, exist_ok=True)
        self.optimizer = None

        # load/define networks
        self.net = CombinedMeshClassifier(opt['network_opt'])
        init_opt = opt['net_init_opt']
        networks.init_net(self.net, init_opt['init_type'], init_opt['init_gain'], self.gpu_ids)
        self.net.train(self.is_train)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss = None

        if self.is_train:
            lr_schedule_opt = opt['lr_schedule_opt']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr_schedule_opt['lr'], betas=(0.9, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, lr_schedule_opt)
            print_network(self.net)

        if not self.is_train or opt['continue_train']:
            self.load_network(opt['which_epoch'])

    def forward(self, data_batch):
        out = self.net(data_batch['pointcloud'], data_batch['edge_features'], data_batch['mesh'])
        return out

    def backward(self, out, data_batch):
        self.loss = self.criterion(out, data_batch['label'])
        self.loss.backward()

    def optimize_parameters(self, data_batch):
        self.optimizer.zero_grad()
        out = self.forward(data_batch)
        self.backward(out, data_batch)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = os.path.join(self.save_dir, save_filename)
        if not os.path.exists(load_path):
            return  # do nothing; let the net be as it is

        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        if self.gpu_ids and len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, data_batch):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward(data_batch)
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = data_batch['label']
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification"""
        correct = pred.eq(labels).sum()
        return correct