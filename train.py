import os
from os.path import dirname
import tqdm
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import importlib
import argparse
from datetime import datetime
from pytz import timezone
import shutil

from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from models.layers import Hourglass

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('-p', '--pretrained_model', type=str, help='path to pretrained model')
    args = parser.parse_args()
    return args

def reload(config):
    """
    Load model's parameters from a specified pretrained model file or continue from a checkpoint.
    The function will also reinitialize the optimizer.
    """
    opt = config['opt']

    if opt.pretrained_model:  # Check if pretrained model path is provided
        if os.path.isfile(opt.pretrained_model):
            print("=> loading pretrained model '{}'".format(opt.pretrained_model))
            checkpoint = torch.load(opt.pretrained_model)
            state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
            config['inference']['net'].load_state_dict(state_dict, strict=False)
            
            # Reinitialize the optimizer
            config['train']['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['inference']['net'].parameters()), lr=config['train']['learning_rate'])
        else:
            print("=> no pretrained model found at '{}'".format(opt.pretrained_model))
            exit(0)

    elif opt.continue_exp:  # Fallback to continue_exp if no pretrained model path is provided
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)
            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0


    
def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')

def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')
    # eric override
    resume_file = '/content/drive/MyDrive/point_localization/exps/checkpoint.pt'
    
    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train(train_func, train_loader, valid_loader, config, post_epoch=None):
    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            loader = train_loader if phase == 'train' else valid_loader
            num_step = len(loader)
            print('start', phase, config['opt'].exp)

            show_range = tqdm.tqdm(range(num_step), total=num_step, ascii=True)
            for i in show_range:
                images, heatmaps = next(iter(loader))
                datas = {'imgs': images, 'heatmaps': heatmaps}
                outs = train_func(i, config, phase, **datas)

        config['train']['epoch'] += 1
        save(config)


def init():
    opt = parse_command_line()
    task = importlib.import_module('task.heart')
    exp_path = os.path.join('exp', opt.exp)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    func = task.make_network(config)
    reload(config)

    train_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Train'
    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    heatmap_res = config['train']['output_res']
    # Initialize your CoordinateDataset and DataLoader here
    im_sz = config['inference']['inp_dim']
    
    train_dataset = CoordinateDataset(root_dir=train_dir, im_sz=im_sz, output_res=heatmap_res, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True, num_workers=4)

    valid_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, output_res=heatmap_res, augment=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config['train']['batchsize'], shuffle=False, num_workers=4)

    return func, train_loader, valid_loader, config

def main():
    func, train_loader, valid_loader, config = init()
    train(func, train_loader, valid_loader, config)
    print(datetime.now(timezone('PST')))

if __name__ == '__main__':
    main()