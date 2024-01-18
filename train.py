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
import copy

from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from models.layers import Hourglass


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='heart', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('-p', '--pretrained_model', type=str, help='path to pretrained model')
    parser.add_argument('-o', '--only10', type=bool, default=False, help='only use 10 images')
    parser.add_argument('-w', '--use_wandb', type=bool, default=False, help='log in wandb')
    args = parser.parse_args()
    return args

sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'min': 0.00002,
            'max': 0.002
        },
        'batch_size': {
            'values': [16, 32]
        },
        # Add other hyperparameters here
    }
}

    
def reload(config):
    """
    Load model's parameters and optimizer state from a checkpoint.
    """
    opt = config['opt']
    #resume = os.path.join('exp', opt['continue_exp'])
    # resume = '/content/drive/MyDrive/point_localization/exps'
    # if config['opt']['continue_exp'] is not None:  # don't overwrite the original exp I guess ??
    #     resume = os.path.join(resume, config['opt']['continue_exp'])
    # else:
    #     resume = os.path.join(resume, config['opt']['exp'])
    # lr_, bs_, = config['train']['learning_rate'], config['train']['batch_size']
    # ###################################################################
    # resume_file = os.path.join(resume, f'checkpoint_{lr_}_{bs_}.pt')
    # eric: make sure wandb isn't running all the same hyperparams across different runs
    ###################################################################
    # resume_file = '/content/drive/MyDrive/point_localization/exps/checkpoint.pt'
    #resume_file = '/content/drive/MyDrive/point_localization/exps/hg2_real/checkpoint_2.133e-05_8.pt'
    if opt['pretrained_model'] is not None:
        resume_file = opt['pretrained_model']

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
    # resume = os.path.join('exp', config['opt']['exp'])
    resume = '/content/drive/MyDrive/point_localization/exps'
    if config['opt']['continue_exp'] is not None:  # don't overwrite the original exp I guess ??
        resume = os.path.join(resume, config['opt']['continue_exp'])
    else:
        resume = os.path.join(resume, config['opt']['exp'])
    lr_, bs_, = config['train']['learning_rate'], config['train']['batch_size']
    resume_file = os.path.join(resume, f'checkpoint_{lr_}_{bs_}.pt')
    # resume_file = '/content/drive/MyDrive/point_localization/exps/checkpoint.pt'
    
    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print(f'=> save checkpoint at {resume_file}')

def train(train_func, config, post_epoch=None):
    
    batch_size = config['train']['batch_size']  # Example of using a hyperparameter

    train_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Train'
    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    heatmap_res = config['train']['output_res']
    # Initialize your CoordinateDataset and DataLoader here
    im_sz = config['inference']['inp_dim']
    train_dataset = CoordinateDataset(root_dir=train_dir, im_sz=im_sz,\
            output_res=heatmap_res, augment=True, only10=config['opt']['only10'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #train_loader = DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True, num_workers=4)

    valid_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz,\
            output_res=heatmap_res, augment=False, only10=config['opt']['only10'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        for phase in ['train', 'valid']:
            if phase == 'valid' and config['train']['epoch'] % 9 != 0: continue # only validate every 4 epochs
            loader = train_loader if phase == 'train' else valid_loader
            num_step = len(loader)

            #########################
            # print(type(config['opt']))
            # print(config['opt'])

            print('start', phase, config['opt']['exp'])

            show_range = tqdm.tqdm(range(num_step), total=num_step, ascii=True)
            for i in show_range:
                images, heatmaps = next(iter(loader))
                datas = {'imgs': images, 'heatmaps': heatmaps}
                outs = train_func(i, config, phase, **datas)
                if config['opt']['use_wandb'] and phase == 'train':
                    wandb.log({"epoch": config['train']['epoch'], "loss": outs["loss"].item(),\
                               "learning_rate": config['train']['learning_rate'], "batch_size": config['train']['batch_size']})

        config['train']['epoch'] += 1
        save(config)


def init(opt):
    # opt is now passed as an argument
    task = importlib.import_module('task.heart')
    config = task.__config__

    opt_dict = vars(opt)

    config['opt'] = opt_dict
    config['train']['epoch'] = 0
    return task, config

def train_with_wandb(task, config):
    config_copy = copy.deepcopy(config)
    config_copy['train']['epoch'] = 0
    wandb.init(config=config_copy)

    # Manually update the learning rate and batch size from wandb.config
    config_copy['train']['learning_rate'] = wandb.config.get('learning_rate', config_copy['train']['learning_rate'])
    config_copy['train']['batch_size'] = wandb.config.get('batch_size', config_copy['train']['batch_size'])

    print(f"Updated learning rate: {config_copy['train']['learning_rate']}")
    print(f"Updated batch size: {config_copy['train']['batch_size']}")

    train_func = task.make_network(config_copy)
    reload(config_copy)
    train(train_func, config_copy)
    wandb.finish()


def main():
    opt = parse_command_line()  # Moved to main()
    task, config = init(opt)
    
    if config['opt']['use_wandb']:
        sweep_id = wandb.sweep(sweep_config, project="2hg-hyperparam-sweep")
        wandb.agent(sweep_id, lambda: train_with_wandb(task, config))
    else:
        train_func = task.make_network(config)
        reload(config)
        train(train_func, config)

if __name__ == '__main__':
    main()
