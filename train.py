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
import wandb

from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from models.layers import Hourglass


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('-p', '--pretrained_model', type=str, help='path to pretrained model')
    parser.add_argument('-o', '--only10', type=bool, default=False, help='only use 10 images')
    args = parser.parse_args()
    return args

sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'lr': {
            'min': 0.001,
            'max': 0.1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        # Add other hyperparameters here
    }
}
    
def reload(config):
    """
    Load model's parameters and optimizer state from a checkpoint.
    """
    opt = config['opt']
    #resume = os.path.join('exp', opt.continue_exp)
    resume = '/content/drive/MyDrive/point_localization/exps'
    if config['opt'].continue_exp is not None:  # don't overwrite the original exp I guess ??
        resume = os.path.join(resume, config['opt'].continue_exp)
    else:
        resume = os.path.join(resume, config['opt'].exp)
    resume_file = os.path.join(resume, f'checkpoint_{config.lr}_{config.bs}.pt')
    # resume_file = '/content/drive/MyDrive/point_localization/exps/checkpoint.pt'

    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        
        config['inference']['net'].load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

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
    # resume = os.path.join('exp', config['opt'].exp)
    resume = '/content/drive/MyDrive/point_localization/exps'
    if config['opt'].continue_exp is not None:  # don't overwrite the original exp I guess ??
        resume = os.path.join(resume, config['opt'].continue_exp)
    else:
        resume = os.path.join(resume, config['opt'].exp)
    resume_file = os.path.join(resume, f'checkpoint_{config.lr}_{config.bs}.pt')
    # resume_file = '/content/drive/MyDrive/point_localization/exps/checkpoint.pt'
    
    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train(task, config, post_epoch=None):
    wandb.init(config=config)
    config = wandb.config

    train_func = task.make_network(config, wandb.config)
    reload(config)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0
    batch_size = config.batch_size  # Example of using a hyperparameter

    train_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Train'
    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    heatmap_res = config['train']['output_res']
    # Initialize your CoordinateDataset and DataLoader here
    im_sz = config['inference']['inp_dim']
    train_dataset = CoordinateDataset(root_dir=train_dir, im_sz=im_sz,\
            output_res=heatmap_res, augment=True, only10=config['opt'].only10)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #train_loader = DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True, num_workers=4)

    valid_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz,\
            output_res=heatmap_res, augment=False, only10=config['opt'].only10)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
                if phase != 'inference':
                    wandb.log({"epoch": config['train']['epoch'], "loss": outs["loss"].item(),\
                               "lr": config.lr, "bs": config.bs})

        config['train']['epoch'] += 1
        save(config)

    wandb.finish()

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

    return task, config

# def main():
#     print(datetime.now(timezone('PST')))
#     func, train_loader, valid_loader, config = init()
#     train(func, train_loader, valid_loader, config)

def train_with_wandb(config=None):
    func, config = init()
    train(func, config)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="2hg-hyperparam-sweep")
    wandb.agent(sweep_id, train_with_wandb)
