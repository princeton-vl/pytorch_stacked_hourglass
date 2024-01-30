import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss
import matplotlib.pyplot as plt
import numpy as np

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(imgs)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    # def heatmapLoss.forward(self, pred, gt):
    #     l = ((pred - gt)**2)
    #     l = l.mean(dim=3).mean(dim=2).mean(dim=1)
    #     return l
def calc_loss(self, combined_hm_preds, heatmaps):
    combined_total_loss = []
    combined_basic_loss = []
    combined_focused_loss = []

    for i in range(self.nstack):
        loss_outputs = self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps)
        combined_total_loss.append(loss_outputs["total_loss"])
        combined_basic_loss.append(loss_outputs["basic_loss"])
        combined_focused_loss.append(loss_outputs["focused_loss"])

    # Stack the total, basic, and focused losses separately
    combined_total_loss = torch.stack(combined_total_loss, dim=1)
    combined_basic_loss = torch.stack(combined_basic_loss, dim=1)
    combined_focused_loss = torch.stack(combined_focused_loss, dim=1)

    # Return a dictionary containing the combined losses
    return {
        "combined_total_loss": combined_total_loss,
        "combined_basic_loss": combined_basic_loss,
        "combined_focused_loss": combined_focused_loss
    }

