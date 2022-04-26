import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgcnn.model import get_graph_feature
from meshcnn.models.networks import MResConv, get_norm_args, get_norm_layer
from meshcnn.models.layers.mesh_pool import MeshPool
from meshcnn.models.layers.mesh import Mesh


class DGCNNFeatureExtractor(nn.Module):
    def __init__(self, opt: dict):
        '''
        opt structure:
            'k': int, for k-NN
            'emb_dims': int, used to calc hidden size for various layers
        '''
        super(DGCNNFeatureExtractor, self).__init__()
        self.k = opt['k']
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(opt['emb_dims'])

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, opt['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.feat_dim = opt['emb_dims'] * 2


    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        return x


class MeshCNNFeatureExtractor(nn.Module):
    def __init__(self, opt: dict):
        '''
        opt structure:
            'nf0': int
                num input channels (5 for the usual MeshCNN initial edge features)
                Corresponds to "opt.input_nc" in original code, with no default (inferred from dataset)

            'conv_res': list of ints
                num out channels (i.e. filters) for each meshconv layer
                Corresponds to "opt.ncf" in original code, with default [16, 32, 32]

            'input_res': int
                num input edges (we take only this many edges from each input mesh)
                Corresponds to "opt.ninput_edges" in original code, with default 750

            'pool_res': list of ints
                num edges to keep after each meshpool layer
                Corresponds to "opt.pool_res" in original code, with default [1140, 780, 580] 

            'norm': str, one of ['batch', 'instance', 'group', 'none']
                type of norm layer to use
                Corresponds to "opt.norm" in original code, with default 'batch'

            'num_groups': int
                num of groups for groupnorm
                Corresponds to "opt.num_groups" in original code, with default 16

            'nresblocks': int
                num res blocks in each mresconv
                Corresponds to "opt.resblocks" in original code, with default 0
        '''
        super(MeshCNNFeatureExtractor, self).__init__()
        self.k = [opt['nf0']] + opt['conv_res']
        self.res = [opt['input_res']] + opt['pool_res']

        norm_layer = get_norm_layer(norm_type=opt['norm'], num_groups=opt['num_groups'])
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], opt['nresblocks']))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        self.gp = nn.AvgPool1d(self.res[-1])
        # self.gp = nn.MaxPool1d(self.res[-1])

        self.feat_dim = self.k[-1]

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        return x


class CombinedFeatureExtractor(nn.Module):
    def __init__(self, dgcnn_opt: dict, meshcnn_opt: dict):
        super(CombinedFeatureExtractor, self).__init__()

        self.dgcnn_branch = DGCNNFeatureExtractor(dgcnn_opt)
        self.meshcnn_branch = MeshCNNFeatureExtractor(meshcnn_opt)

        self.feat_dim = self.dgcnn_branch.feat_dim + self.meshcnn_branch.feat_dim


    def forward(self, vertex_input_batch, edge_input_batch, mesh_batch):
        vertex_based_feats = self.dgcnn_branch(vertex_input_batch)
        edge_based_feats = self.meshcnn_branch(edge_input_batch, mesh_batch)

        out = torch.cat([vertex_based_feats, edge_based_feats], dim=-1)
        return out


class CombinedMeshClassifier(nn.Module):
    def __init__(self, opt: dict):
        '''
        opt structure:
            'classifier_opt': dict (nested options), structure:
                'out_num_classes': int, num classes to classify i.e. final layer size of output MLP
                'out_block_hidden_dims': list of ints, hidden layer sizes of output MLP
            'dgcnn_opt': dict (nested options), see DGCNN feat ex for details
            'meshcnn_opt': dict (nested options), see MeshCNN feat ex for details
        '''
        super(CombinedMeshClassifier, self).__init__()

        self.feat_ex = CombinedFeatureExtractor(dgcnn_opt=opt['dgcnn_opt'], meshcnn_opt=opt['meshcnn_opt'])

        classifier_opt = opt['classifier_opt']
        out_layers = []
        in_dim = self.feat_ex.feat_dim
        for hdim in classifier_opt['out_block_hidden_dims']:
            out_layers.extend([
                nn.Linear(in_features=in_dim, out_features=hdim),
                nn.ReLU()
            ])
            in_dim = hdim
        out_layers.append(nn.Linear(in_features=in_dim, out_features=classifier_opt['out_num_classes']))
        self.output_block = nn.Sequential(out_layers)

    def forward(self, vertex_input_batch, edge_input_batch, mesh_batch):
        combined_feats = self.feat_ex(vertex_input_batch, edge_input_batch, mesh_batch)
        out = self.output_block(combined_feats)
        return out