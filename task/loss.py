# import torch

# class HeatmapLoss(torch.nn.Module):
#     """
#     loss for detection heatmap
#     """
#     def __init__(self):
#         super(HeatmapLoss, self).__init__()

#     def forward(self, pred, gt):
#         l = ((pred - gt)**2)
#         l = l.mean(dim=3).mean(dim=2).mean(dim=1)
#         return l ## l of dim bsize


import torch

class HeatmapLoss(torch.nn.Module):
    """
    Loss for detection heatmap with additional weight on the first 2 of 6 points along dim=1,
    focusing only on horizontal (lateral) errors.
    """
    def __init__(self, extra_weight=2.0):
        super(HeatmapLoss, self).__init__()
        self.extra_weight = extra_weight

    def forward(self, pred, gt):
        # Basic loss calculation
        basic_loss = ((pred - gt)**2).mean(dim=3).mean(dim=2).mean(dim=1)

        # Additional loss term for the first 2 points focusing on lateral errors
        # Assuming lateral errors are along the width (last dimension)
        lateral_pred = pred[:, :2].sum(dim=2)  # Summing over the vertical dimension
        lateral_gt = gt[:, :2].sum(dim=2)  # Summing over the vertical dimension
        focused_loss = ((lateral_pred - lateral_gt)**2).mean(dim=2).mean(dim=1) * self.extra_weight

        # Combine the losses
        total_loss = basic_loss + focused_loss
        return total_loss
