import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.layer import *
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    



class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # decomp for gnn
        self.kernel_lst = [15, 25, 61]
        self.gnn_decomp_lst = nn.ModuleList()
        self.linear_decomp_lst = nn.ModuleList()
        for i in range(len(self.kernel_lst)):
            self.linear_decomp_lst.append(nn.Linear(self.seq_len,self.pred_len))
            self.gnn_decomp_lst.append(series_decomp(self.kernel_lst[i]))


        # layer norm
        self.layer_norm = nn.LayerNorm([configs.enc_in, configs.pred_len])


        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

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
        self.gconv_in = len(self.kernel_lst) + 1 + 1# 1
        self.gconv_out = 1
        dropout = 0.3
        propalpha = 0.05
        gcn_depth = 3
        for _ in range(self.num_layer):
            self.gconv1.append(mixprop(self.gconv_in, self.gconv_out, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(self.gconv_in, self.gconv_out, gcn_depth, dropout, propalpha))


        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # print('torch.max(trend_init)', torch.max(trend_init))
        # print('torch.min(trend_init)', torch.min(trend_init))
        # print('torch.max(seasonal_init)', torch.max(seasonal_init))
        # print('torch.min(seasonal_init)', torch.min(seasonal_init))
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        trend_output = self.layer_norm(trend_output)
        adp = self.gc(self.idx)
        # print(seasonal_output.shape)
        # print('torch.max(trend_output)', torch.max(trend_output))
        # print('torch.min(trend_output)', torch.min(trend_output))
        # print('torch.max(seasonal_output)', torch.max(seasonal_output))
        # print('torch.min(seasonal_output)', torch.min(seasonal_output))
        seasonal_lst = []
        for i in range(len(self.kernel_lst)):
            curr_seasonal, _ = self.gnn_decomp_lst[i](x)
            # curr_seasonal, _ = self.decompsition(x)
            curr_seasonal = curr_seasonal.permute(0,2,1)
            curr_seasonal = self.linear_decomp_lst[i](curr_seasonal)
            seasonal_lst.append(curr_seasonal.unsqueeze(1))

        

        # seasonal_output_gconv = seasonal_output.unsqueeze(1)

        seasonal_lst.append(seasonal_output.unsqueeze(1))
        seasonal_lst.append(trend_output.unsqueeze(1))
        # for item in seasonal_lst:
        #     print(item.shape)
        seasonal_output_gconv = torch.cat(seasonal_lst, dim=1).cuda()
        # print(seasonal_output_gconv.shape)
        
        for i in range(self.num_layer):
            seasonal_output_gconv = self.gconv1[i](seasonal_output_gconv, adp)+self.gconv2[i](seasonal_output_gconv, adp.transpose(1,0))
        seasonal_output_gconv = seasonal_output_gconv.squeeze()

        # print('torch.max(trend_output)', torch.max(trend_output))
        # print('torch.min(trend_output)', torch.min(trend_output))
        # print('torch.max(seasonal_output_gconv)', torch.max(seasonal_output_gconv))
        # print('torch.min(seasonal_output_gconv)', torch.min(seasonal_output_gconv))
        # print(seasonal_output.shape)
        # seasonal_output += seasonal_output_gconv
        x = seasonal_output_gconv + trend_output
        # exit(0)
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
