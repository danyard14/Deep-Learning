import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)  # TODO: check if needed

    def forward(self, x):
        """
        :param x: input of shape (batch, sequence_size, num_of_features)
        :return:
        """
        seq_size = x.shape[1]
        output, (h_n, c_n) = self.encoder.forward(x)
        input_z = torch.cat([h_n] * seq_size, axis=0)   # duplicate z T times so it can be used as input for decoder
        output, _ = self.decoder.forward(input_z)
        output = self.linear(output)
        output = output.view((output.shape[1], output.shape[0], -1))

        return output
