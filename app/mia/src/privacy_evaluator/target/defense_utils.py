import torch
import torch.nn as nn
import torch.nn.functional as F

def CrossEntropy_soft(input, target, reduction='mean'):
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def one_hot_embedding(label, num_class, dtype=torch.cuda.FloatTensor):
    device = f"cuda:1"
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    scatter_dim = len(label.size())
    # label_tensor = label.type(torch.cuda.LongTensor).view(*label.size(), -1)
    y_tensor = label.view(*label.size(), -1)
    zeros = torch.zeros(*label.size(), num_class).type(dtype).to(device)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def adjust_learning_rate(optimizer, epoch, gamma, schedule_milestone):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    if epoch in schedule_milestone:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr