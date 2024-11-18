from abc import ABC, abstractmethod

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import torch.nn as nn

class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
class PatchCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logit, label):
        _, _, H, W = logit.shape
        label = label.unsqueeze(-1).unsqueeze(-1).repeat(1,H,W)
        loss = self.ce(logit, label)
        return loss        

def define_criterion(name, label_smoothing):
    if name == "ce":
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif name == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif name == "patchCE":
        criterion = PatchCE()
    return criterion

def define_optimizer(model, name, **kwargs):
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs["lr"], betas=kwargs["betas"], weight_decay=kwargs["weight_decay"])
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"], weight_decay=kwargs['weight_decay'])
    return optimizer

def define_scheduler(optimizer, name, **kwargs):
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=kwargs["T_0"], T_mult=kwargs["T_mult"], eta_min=kwargs["eta_min"])
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"])
    return scheduler

#### Fast AWP ####
# AWP reference: https://www.kaggle.com/code/junkoda/fast-awp
class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self, input_ids, attention_mask, y, criterion):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])