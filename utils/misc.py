import torch
import numpy as np
import importlib

# Helpers when setting up training

def importNet(net):
    t = net.split('.')
    path, name = '.'.join(t[:-1]), t[-1]
    module = importlib.import_module(path)
    return eval('module.{}'.format(name))

def make_input(t, requires_grad=False, need_cuda = True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.cuda()
    return inp

def make_output(x):
    if not (type(x) is list):
        return x.cpu().data.numpy()
    else:
        return [make_output(i) for i in x]
