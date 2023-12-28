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

__config__ = {
    'data_provider': 'data.MPII.dp',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 2,
        'inp_dim': 256,
        'oup_dim': 6,
        'num_parts': 6,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958, ## number of val examples used. entire set is 2958
        'train_num_eval': 300, ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 16,
        'input_res': 256,
        'output_res': 64,
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 1e-3,
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
            loss = self.calc_loss(**labels, combined_hm_preds=combined_hm_preds)
            return list(combined_hm_preds) + list([loss])

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    def calc_loss(*args, **kwargs):
        return poseNet.calc_loss(*args, **kwargs)
    
    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(**config)
    forward_net = poseNet.cuda()
    if torch.cuda.device_count() > 1:
        forward_net = DataParallel(forward_net)
    config['net'] = Trainer(forward_net, configs['inference']['keys'], calc_loss)
    
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'])

    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp=='pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

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

        if phase != 'inference':
            result = net(inputs['imgs'], **{i:inputs[i] for i in inputs if i!='imgs'})
            num_loss = len(config['train']['loss'])

            losses = {i[0]: result[-num_loss + idx]*i[1] for idx, i in enumerate(config['train']['loss'])}
                        
            loss = 0
            toprint = '\n{}: '.format(batch_id)
            for i in losses:
                loss = loss + torch.mean(losses[i])

                my_loss = make_output( losses[i] )
                my_loss = my_loss.mean()

                if my_loss.size == 1:
                    toprint += ' {}: {}'.format(i, format(my_loss.mean(), '.8f'))
                else:
                    toprint += '\n{}'.format(i)
                    for j in my_loss:
                        toprint += ' {}'.format(format(j.mean(), '.8f'))
            logger.write(toprint)
            logger.flush()
            
            if phase == 'train':
                optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if batch_id == config['train']['decay_iters']:
                ## decrease the learning rate after decay # iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['train']['decay_lr']
            
            return None
        else:
            out = {}
            net = net.eval()
            result = net(**inputs)
            if type(result)!=list and type(result)!=tuple:
                result = [result]
            out['preds'] = [make_output(i) for i in result]
            return out
    return make_train
