import torch.nn as nn
from .encoder import LSTM_encoder, TCN_encoder, Transformer_encoder, CPC_encoder


class ContrastADModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['encoder'] == 'lstm':
            self.encoder_model = LSTM_encoder(input_size=config['input_size'],
                                              hidden_dim=config['hidden_dim'],
                                              batch_size=config['batch_size'],
                                              num_layers=config['num_layers'],
                                              dropout_rate=config['dropout_rate'],
                                              direction=config['direction'],
                                              seed=config['seed'])
        elif config['encoder'] == 'tcn':
            self.encoder_model = TCN_encoder(input_dim=config['input_size'],
                                             hidden_size=config['hidden_dim'],
                                             kernel_size=config['tcn_kernel_size'],
                                             dropout=config['dropout_rate'],
                                             seed=config['seed'])
        elif config['encoder'] == 'transformer':
            self.encoder_model = Transformer_encoder(input_dim=config['input_size'],
                                                     hidden_dim=config['hidden_dim'],
                                                     seed=config['seed'])
        elif config['encoder'] == 'cpc':
            self.encoder_model = CPC_encoder(input_size=config['input_size'],
                                             enc_emb_size=config['hidden_dim']*4,
                                             ar_emb_size=config['hidden_dim'],
                                             n_prediction_steps=12,
                                             seed=config['seed'])
        else:
            raise NotImplementedError('No implementation of encoder')

    def forward(self, x):
        representation = self.encoder_model(x)

        return representation
