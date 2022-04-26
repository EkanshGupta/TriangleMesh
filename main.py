'''
Usage: main.py [train|test] CONFIG_YAML_PATH
'''

import os
import sys
import pickle
import copy
import math
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model_wrapper import ClassifierWrapper
from data import DataLoader
from meshcnn.util.writer import Writer


def read_yaml_config(yaml_config_path):
  if yaml_config_path is None:
    return None
  with open(yaml_config_path, 'r') as yml_file:
    cfg = yaml.load(yml_file, Loader=yaml.FullLoader)

  # resolve paths
  def resolve_paths(obj):
    if isinstance(obj, str):
      return (
        os.path.abspath(os.path.join(
          os.path.dirname(yaml_config_path), obj
        ))
        if ('/' in obj or '\\' in obj) and not os.path.isabs(obj)
        else obj
      )
    elif isinstance(obj, dict):
      for k,v in obj.items():
        obj[k] = resolve_paths(v)
      return obj
    elif hasattr(obj, '__iter__'):
      for i, x in enumerate(obj):
        obj[i] = resolve_paths(x)
      return obj
    else:
      return obj

  cfg = resolve_paths(cfg)
  return cfg


def test(opt: dict, epoch=-1):
    print('Running Test')

    dataloader_opt = opt['dataloader_opt']
    model_wrapper_opt = opt['model_wrapper_opt']

    dataloader = DataLoader('test', dataloader_opt)
    print(f"data num classes: {dataloader.dataset.nclasses}")
    model = ClassifierWrapper(False, model_wrapper_opt)
    writer = Writer(False, model_wrapper_opt['checkpoints_dir'], model_wrapper_opt['expt_name'])
    # test
    writer.reset_counter()
    for i, data in enumerate(dataloader):
        ncorrect, nexamples = model.test(data)
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


def train(opt: dict):
    dataloader_opt = opt['dataloader_opt']
    model_wrapper_opt = opt['model_wrapper_opt']
    recording_opt = opt['recording_opt']

    dataloader = DataLoader('train', dataloader_opt)
    dataset_size = len(dataloader)
    print('#training meshes = %d' % dataset_size)

    model = ClassifierWrapper(True, model_wrapper_opt)
    writer = Writer(True, model_wrapper_opt['checkpoints_dir'], model_wrapper_opt['expt_name'])

    lr_schedule_opt = model_wrapper_opt['lr_schedule_opt']
    batch_size = dataloader_opt['batch_size']

    total_steps = 0
    for epoch in range(lr_schedule_opt['epoch_count'], lr_schedule_opt['niter'] + lr_schedule_opt['niter_decay'] + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_steps % recording_opt['print_freq'] == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += batch_size
            epoch_iter += batch_size
            model.optimize_parameters(data)

            if total_steps % recording_opt['print_freq'] == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % recording_opt['save_latest_freq'] == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % recording_opt['save_epoch_freq'] == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, lr_schedule_opt['niter'] + lr_schedule_opt['niter_decay'], time.time() - epoch_start_time))
        model.update_learning_rate()

        if epoch % recording_opt['run_test_freq'] == 0:
            acc = test(epoch=epoch, opt=opt)
            writer.plot_acc(acc, epoch)

    writer.close()


if __name__ == '__main__':
    mode = sys.argv[1]
    config_yaml_path = sys.argv[2]
    opt_full = read_yaml_config(config_yaml_path)

    if mode == 'train':
        train(opt_full)
    elif mode == 'test':
        test(opt_full)
    else:
        print(f'Invalid mode: {mode}')