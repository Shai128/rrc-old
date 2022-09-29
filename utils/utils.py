import functools
import os

import torch
from torch import nn
import dill
from collections import OrderedDict

from helper import create_folder_if_it_doesnt_exist


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)



def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def has_state(dataset_name, model_name, step, epoch):
    path = get_checkpoint_path(dataset_name, model_name, step, epoch)
    return os.path.exists(path)


def get_checkpoint_path(dataset_name, model_name, step, epoch):
    checkpoint_path = f"{get_checkpoint_dir(dataset_name, model_name)}/{model_name}_epoch={epoch}.pth"
    return checkpoint_path

def get_checkpoint_dir(dataset_name, model_name):
    checkpoint_dir = f"saved_models/{dataset_name}"
    return checkpoint_dir

def save_ckpt(dataset_name, model_name, step, epoch, model, optimizer, scheduler):
    """Save checkpoint"""
    create_folder_if_it_doesnt_exist(get_checkpoint_dir(dataset_name, model_name))
    checkpoint_path = get_checkpoint_path(dataset_name, model_name, step, epoch)
    torch.save({
        'step': step,
        'epoch': epoch,
        'scheduler': scheduler.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        checkpoint_path, pickle_module=dill)

def load_ckpt(dataset_name, model_name, model, step, epoch, optimizer=None, scheduler=None):
    """
    Load checkpoint.
    """
    checkpoint_path = get_checkpoint_path(dataset_name, model_name, step, epoch)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, pickle_module=dill)
        model_state_dict_keys = model.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

        if all(key.startswith('module.') for key in model_state_dict_keys):
            model.module.load_state_dict(checkpoint_state_dict_noprefix)
        else:
            model.load_state_dict(checkpoint_state_dict_noprefix)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.__setattr__('last_epoch', checkpoint['step'])
        del checkpoint
        torch.cuda.empty_cache()
