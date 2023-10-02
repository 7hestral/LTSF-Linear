import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.layer import *
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
         # MTGNN setup
        num_nodes = configs.enc_in
        subgraph_size = min(num_nodes, 20)
        node_dim = 40
        tanhalpha = 3
        static_feat = None
        self.idx = torch.arange(num_nodes).cuda()
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, alpha=tanhalpha, static_feat=static_feat)
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.num_layer = 1
        self.gconv_in = 1
        self.gconv_out = 1
        dropout = 0.3
        propalpha = 0.05
        gcn_depth = 3
        for _ in range(self.num_layer):
            self.gconv1.append(mixprop(self.gconv_in, self.gconv_out, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(self.gconv_in, self.gconv_out, gcn_depth, dropout, propalpha))

    def forward(self, x):
        adp = self.gc(self.idx)
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1))# .permute(0,2,1)
        
        x = x.unsqueeze(1)
        # print(seasonal_output.shape)
        for i in range(self.num_layer):
            x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
        x = x.squeeze()

        
        x = x.permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]