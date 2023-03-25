from typing import List

import torch


class Network(torch.nn.Module):

    def __init__(self, cfg, time_series_length:int, device:str='cpu'):
        super(Network, self).__init__()

        self.input_size = cfg.input_size
        self.latent_size = cfg.latent_size
        self.output_size = cfg.output_size
        self.encoder_num_units = cfg.encoder_num_units
        self.decoder_num_units = cfg.decoder_num_units
        self.time_series_length = time_series_length
        self.device = device
        
        # Set up encoder
        self.encoder_layers = [
            torch.nn.Linear(self.input_size, self.encoder_num_units[0]),
            torch.nn.ReLU()
        ]
        for i in range(len(self.encoder_num_units)-1):
            self.encoder_layers.extend([
                torch.nn.Linear(self.encoder_num_units[i], self.encoder_num_units[i+1]),
                torch.nn.ReLU()
            ])
        self.encoder_layers.extend([
            torch.nn.Linear(self.encoder_num_units[-1], self.latent_size)
        ])
        self.encoder_layers = torch.nn.Sequential(*self.encoder_layers)

        # Set up time evolution network
        self.time_evolution_layer = torch.nn.Linear(self.latent_size, self.latent_size)
        self.time_evolution_layer.weight.data.fill_(1.0)
        #self.time_evolution_layer.weight.requires_grad_(False)
        self.time_evolution_layer.bias.data.fill_(0.0)

        # Set up decoder
        self.decoder_layers = [
            torch.nn.Linear(self.latent_size+1, self.decoder_num_units[0]),
            torch.nn.ReLU()
        ]
        for i in range(len(self.decoder_num_units)-1):
            self.decoder_layers.extend([
                torch.nn.Linear(self.decoder_num_units[i], self.decoder_num_units[i+1]),
                torch.nn.ReLU()
            ])
        self.decoder_layers.extend([
            torch.nn.Linear(self.decoder_num_units[-1], self.output_size)
        ])
        self.decoder_layers = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, input, ecc):
        latent_code = self.encoder_layers(input[:,:self.input_size].float())
        output = []
        for i in range(self.time_series_length):
            latent_code = self.time_evolution_layer(latent_code)
            latent_code_with_ecc = torch.cat([latent_code, ecc], dim=-1)
            output.append(self.decoder_layers(latent_code_with_ecc))
        return torch.cat(output, dim=1)
