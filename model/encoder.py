import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .utils.cpc import CPCNetwork
from .utils.tstcc import Seq_Transformer


class LSTM_encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, batch_size, num_layers, dropout_rate, direction, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.direction = direction
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout_rate,
                               batch_first=True,
                               bidirectional=True if direction == 2 else False)

    def forward(self, x):
        encoder_hidden = self.init_hidden(x.device)
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden)
        representation = encoder_hidden[0]
        if self.direction == 2:
            representation = torch.mean(representation, dim=0)
        return torch.squeeze(representation)

    def init_hidden(self, device):
        """
        Returns: two zero matrix for initial hidden state and cell state
        """
        h_s, c_s = torch.zeros(self.num_layers * self.direction, self.batch_size, self.hidden_dim,
                               device=device), \
                   torch.zeros(self.num_layers * self.direction, self.batch_size, self.hidden_dim,
                               device=device)

        return h_s, c_s


# Adapted from: https://github.com/locuslab/
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, kernel_size, dropout, seed):
        super().__init__()
        num_channels = [30] * 8
        torch.manual_seed(seed)
        self.tcn = TemporalConvNet(input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], hidden_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


class Transformer_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = Seq_Transformer(patch_size=input_dim, dim=hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, x):
        return self.encoder(x)


class CPC_encoder(nn.Module):
    def __init__(self, input_size, enc_emb_size, ar_emb_size, n_prediction_steps, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = CPCNetwork(input_size, enc_emb_size, ar_emb_size, n_prediction_steps)

    def forward(self, x):
        z, c = self.encoder.get_embeddings(x, x.device)

        return c.reshape(x.shape[0], c.shape[-1])
