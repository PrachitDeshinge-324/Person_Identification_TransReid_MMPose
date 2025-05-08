# OpenGait utility functions for inference
# Copied from OpenGait/opengait/utils/common.py

import os
import numpy as np
import torch
import torch.nn.functional as F


def get_msg_mgr():
    from ..core.msg_mgr import MsgMgr
    return MsgMgr()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank():
    return int(os.environ.get("RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def is_dist():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def is_main_process():
    return get_rank() == 0


def synchronize():
    if not is_dist():
        return
    torch.distributed.barrier()


def reduce_tensor(tensor, world_size=None):
    if world_size is None:
        world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        reduced_tensor = tensor.clone()
        torch.distributed.reduce(reduced_tensor, dst=0)
    return reduced_tensor / world_size


def all_gather(data):
    if not is_dist():
        return [data]
    world_size = get_world_size()
    gathered_data = [None] * world_size
    torch.distributed.all_gather_object(gathered_data, data)
    return gathered_data


def all_reduce(data, op=torch.distributed.ReduceOp.SUM):
    if not is_dist():
        return data
    with torch.no_grad():
        reduced_data = data.clone()
        torch.distributed.all_reduce(reduced_data, op=op)
    return reduced_data


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)


def load_checkpoint(filepath, model, optimizer=None):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch):
    """Sets the learning rate to the initial LR decayed by lr_decay_rate every lr_decay_epoch epochs"""
    lr = optimizer.param_groups[0]['lr'] * (lr_decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, warmup_iter, current_iter, base_lr):
    """Warmup learning rate"""
    lr = base_lr * (current_iter / warmup_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_annealing_lr(optimizer, current_iter, max_iter, base_lr, min_lr=0):
    """Cosine annealing learning rate"""
    lr = min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model