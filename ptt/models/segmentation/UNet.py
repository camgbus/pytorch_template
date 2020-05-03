import os
import torch
import torch.nn as nn
from src.utils.pytorch.pytorch_load_restore import load_model_state

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )    

class UNet(nn.Module):

    def __init__(self, n_class, n_input_layers, n_input_channels, weights_file_path=None):
        super().__init__()

        self.dconv_down1 = double_conv(n_input_channels, n_input_layers)
        self.dconv_down2 = double_conv(n_input_layers, n_input_layers*2)
        self.dconv_down3 = double_conv(n_input_layers*2, n_input_layers*4)
        self.dconv_down4 = double_conv(n_input_layers*4, n_input_layers*8)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(n_input_layers*4 + n_input_layers*8, n_input_layers*4)
        self.dconv_up2 = double_conv(n_input_layers*2 + n_input_layers*4, n_input_layers*2)
        self.dconv_up1 = double_conv(n_input_layers*2 + n_input_layers, n_input_layers)
        
        self.conv_last = nn.Conv2d(n_input_layers, n_class, 1)

        self.initialize(weights_file_path)

    def get_layer(self, key):
            return getattr(self, key)

    def initialize(self, weights_file_path=None):
        """
        Tries to restore a previous model. If no model is found but a file name
        if provided, the model is saved.
        """
        if weights_file_path is not None:
            self.load_state_dict(torch.load(weights_file_path))
            print('Parameters {} were restored'.format(weights_file_path))


    def preprocess_input(self, x):
        return x

    def predict(self, x):
        inputs = self.preprocess_input(x)
        outputs = self.forward(inputs)
        outputs = torch.sigmoid(outputs)
        return outputs
        
    def forward(self, x):
        conv1 = self.get_layer('dconv_down1')(x)
        #conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.get_layer('dconv_down2')(x)
        #conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.get_layer('dconv_down3')(x)
        #conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.get_layer('dconv_down4')(x)
        #x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.get_layer('dconv_up3')(x)
        #x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.get_layer('dconv_up2')(x)
        #x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.get_layer('dconv_up1')(x)
        #x = self.dconv_up1(x)
        
        out = self.get_layer('conv_last')(x)
        #out = self.conv_last(x)
        
        return out