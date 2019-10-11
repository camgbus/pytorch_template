################################################################################
# Pretrained InceptionV3 model provided by torchvision
################################################################################

import torch.nn as nn
import torchvision.models as models

def InceptionV3(model_config):
    model = models.inception_v3(pretrained=model_config['pretrained'])
    if model_config['freeze_params']:
        for params in model.parameters():
            params.requires_grad = False
        # Unfreeze the last layers 
        unfreeze = False
        for name, named_child in model.named_children():
            if unfreeze:
                for params in named_child.parameters():
                    params.requires_grad = True
            if 'Conv2d_4a_3x3' in name:
                unfreeze = True
    # Replace final layer with one with the right nr. of outputs
    model.fc = nn.Linear(model.fc.in_features, model_config['nr_outputs'])
    return model