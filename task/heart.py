"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from utils.misc import make_input, make_output, importNet

import matplotlib.pyplot as plt

__config__ = {
    'data_provider': 'data.MPII.dp',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 4,
        'inp_dim': 300,
        'oup_dim': 6,
        'num_parts': 6,
        'increase': 0,
        'keys': ['imgs']
    },

    'train': {
        'epoch_num': 11,
        'learning_rate': .00002,#02133,
        'batch_size': 4,
        'input_res': 300,
        'output_res': 75,
        'e': 1000,
        'valid_iters': 10,
        'max_num_people' : 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 100000,
        'decay_lr': 2e-4,
        'num_workers': 2,
        'use_data_loader': True,
    },
}

class Trainer(nn.Module):
    """
    The wrapper module that will behave differetly for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]

        if not self.training:
            return self.model(imgs, **inps)
        else:
            combined_hm_preds = self.model(imgs, **inps)
            if type(combined_hm_preds)!=list and type(combined_hm_preds)!=tuple:
                combined_hm_preds = [combined_hm_preds]
            true_heatmaps = labels['heatmaps']
            loss = self.calc_loss(combined_hm_preds, true_heatmaps)

            return list(combined_hm_preds) + list([loss])

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    def calc_loss(*args, **kwargs):
        return poseNet.calc_loss(*args, **kwargs)
        # loss = poseNet.calc_loss(*args, **kwargs)
        # return {"total_loss": loss}
    
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(**config)
    forward_net = poseNet.cuda()
    if torch.cuda.device_count() > 1:
        forward_net = DataParallel(forward_net)
    config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss)

    learning_rate = train_cfg['learning_rate']
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['net'].parameters()), lr=learning_rate)
    
    ## optimizer, experiment setup
    exp_path = '/content/drive/MyDrive/point_localization/exps'
    if configs['opt']['continue_exp'] is not None:  # don't overwrite the original exp I guess ??
        exp_path = os.path.join(exp_path, configs['opt']['continue_exp'])
    else:
        exp_path = os.path.join(exp_path, configs['opt']['exp'])

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    # **inputs = {'imgs': images, 'heatmaps': heatmaps}
    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass #for last input, which is a string (id_)
                
        net = config['inference']['net']
        config['batch_id'] = batch_id

        net = net.train()

        # When in validation phase put batchnorm layers in eval mode
        # to prevent running stats from getting updated.
        if phase == 'valid':
            for module in net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
#  ERIC: which phase is being used??
        if phase != 'inference':
            result = net(inputs['imgs'], **{i:inputs[i] for i in inputs if i!='imgs'})
            # Assuming result[0] are the predictions and result[1] are the losses
            combined_hm_preds = result[:-1]
            loss_dict = result[-1]

            combined_total_loss = loss_dict["combined_total_loss"]
            # combined_basic_loss = loss_dict["combined_basic_loss"]
            # combined_focused_loss = loss_dict["combined_focused_loss"]

            # toprint2 = f"a: {combined_basic_loss}"
            # toprint3 = f"b: {combined_focused_loss}"
            # toprint = f"c: {combined_total_loss}"
            # logger.write(toprint)
            # logger.write(toprint2)
            # logger.write(toprint3)
            # logger.flush()

            # Aggregate loss across all stacks
            total_loss = combined_total_loss.mean()
            # basic_loss = combined_basic_loss.mean()
            # focused_loss = combined_focused_loss.mean()

            # Logging
            toprint = f'\n{batch_id}: Total Loss: {total_loss.item():.8f}\n'
            for stack_idx in range(combined_total_loss.shape[1]):
                stack_loss = combined_total_loss[:, stack_idx].mean()
                toprint += f'Stack {stack_idx} Loss: {stack_loss.item():.8f}\n'

            logger.write(toprint)
            logger.flush()

            # Backpropagation and optimization step
            if phase == 'train':
                optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Learning rate decay
            if batch_id == config['train']['decay_iters']:
                for param_group in optimizer.param_groups:
                    param_group['learning_rate'] = config['train']['decay_lr']
            return {"total_loss": total_loss,
                    # "basic_loss": basic_loss,
                    # "focused_loss": focused_loss,
                    "predictions": combined_hm_preds}

        else:
            out = {}
            net = net.eval()
            result = net(**inputs)
            if type(result)!=list and type(result)!=tuple:
                result = [result]
            out['preds'] = [make_output(i) for i in result]
            return out
    return make_train
