import torch
class HeatmapLoss(torch.nn.Module):
    def __init__(self, extra_weight=2.0):
        super(HeatmapLoss, self).__init__()
        self.extra_weight = extra_weight

    def forward(self, pred, gt):
        basic_loss = ((pred - gt)**2).mean(dim=3).mean(dim=2).mean(dim=1)
        lateral_pred = pred[:, :2, :, -1]
        lateral_gt = gt[:, :2, :, -1]
        focused_loss = ((lateral_pred - lateral_gt)**2).mean() * self.extra_weight
        total_loss = basic_loss + focused_loss

        return {'total_loss': total_loss, 'basic_loss': basic_loss, 'focused_loss': focused_loss}
