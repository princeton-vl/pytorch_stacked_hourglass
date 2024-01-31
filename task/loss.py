import torch
import os

class HeatmapLoss(torch.nn.Module):
    def __init__(self, extra_weight=0.2):
        super(HeatmapLoss, self).__init__()
        self.extra_weight = extra_weight

    def forward(self, pred, gt):
        #logger.write(f'pred.shape: {pred.shape}')  # (4,6,75,75)
        # exp_path = '/content/drive/MyDrive/point_localization/exps/hg8_1ch_300x300_lat_loss8'
        # if not os.path.exists(exp_path):
        #     os.mkdir(exp_path)
        #logger = open(os.path.join(exp_path, 'log'), 'a+')
        #logger.flush()
        
        # Basic loss calculation remains the same, resulting in a tensor of shape [batch_size]
        basic_loss = ((pred - gt)**2).mean(dim=3).mean(dim=2).mean(dim=1)

        # Extract lateral slices for pred and gt, same as before
        lateral_pred = pred[:, :2, :, -1]
        lateral_gt = gt[:, :2, :, -1]

        # Modify focused loss calculation to maintain the batch dimension
        # Calculate squared difference and then mean across the channel and height (since width is already reduced by slicing)
        focused_loss = ((lateral_pred - lateral_gt)**2).mean(dim=2).mean(dim=1) * self.extra_weight

        # total_loss calculation needs to be adjusted since basic_loss and focused_loss now have the same shape [batch_size]
        total_loss = basic_loss + focused_loss

        return {'total_loss': total_loss, 'basic_loss': basic_loss, 'focused_loss': focused_loss}