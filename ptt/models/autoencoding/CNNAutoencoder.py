from src.models.autoencoding.autoencoder import Autoencoder
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(Autoencoder):    
    def __init__(self, config, in_channels, img_size):
        super().__init__(config=config, in_channels=in_channels, img_size=img_size)

        self.encoder_conv_1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=3, padding=1)
        self.layers['encoder_conv_1'] = self.encoder_conv_1
        self.operations_after['encoder_conv_1'] = [F.relu, lambda x : F.max_pool2d(x, (2, 2), stride=2)]

        self.encoder_conv_2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.layers['encoder_conv_2'] = self.encoder_conv_2
        self.operations_after['encoder_conv_2'] = [F.relu, lambda x : F.max_pool2d(x, (2, 2), stride=1)]

        self.first_decoder_layer_name = 'decoder_conv_1'

        self.decoder_conv_1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.layers['decoder_conv_1'] = self.decoder_conv_1
        self.operations_after['decoder_conv_1'] = [F.relu]

        self.decoder_conv_2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1)
        self.layers['decoder_conv_2'] = self.decoder_conv_2
        self.operations_after['decoder_conv_2'] = [F.relu]

        self.decoder_conv_3 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=1)
        self.layers['decoder_conv_3'] = self.decoder_conv_3
        self.operations_after['decoder_conv_3'] = [F.tanh]
 