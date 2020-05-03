import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoding.autoencoder import Autoencoder

class LinearAutoencoder(Autoencoder):    
    def __init__(self, config, in_channels=1, img_size=(320, 320), hidden_dim=100):

        super().__init__(config=config, in_channels=in_channels, img_size=img_size)

        input_dim = in_channels*img_size[0]*img_size[1]

        self.encoder_1 = nn.Linear(input_dim, hidden_dim)
        self.layers['encoder_1'] = self.encoder_1
        self.operations_after['encoder_1'] = [F.relu]

        self.first_decoder_layer_name = 'decoder_1'

        self.decoder_1 = nn.Linear(hidden_dim, input_dim)
        self.layers['decoder_1'] = self.decoder_1
        self.operations_after['decoder_1'] = [F.relu]

    def preprocess_input(self, x):
        # Linearize
        return self.flatten(x)
