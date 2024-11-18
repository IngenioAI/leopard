import timm
import torch
import torch.nn as nn

# netwrok parameter initialization.
def he_init(module):
    if isinstance(module, nn.Conv2d): 
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def define_net(model_name, no_pretrained, n_classes, drop_rate=0.0):
    if model_name in timm.list_models():
        model = timm.create_model(model_name, pretrained=not no_pretrained, num_classes=n_classes, drop_rate=drop_rate)
    else:
        raise NotImplementedError(F"model name {model_name} is not in timm")
    # if "CLIP" in model_name:
    #     clip_load(model_name, model)    
    if no_pretrained:
        model.apply(he_init)
    return model
