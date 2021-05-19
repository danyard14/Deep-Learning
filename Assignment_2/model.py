import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, T, classify=False):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.classify = classify
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size) if not classify \
            else nn.Linear(self.hidden_size * T, output_size)   # TODO: check if multiply by T is correct

    def forward(self, x):
        """
        :param x: input of shape (batch X sequence_size X num_of_features)
        :return:
        """
        seq_size = x.shape[1]
        output, _ = self.encoder.forward(x)
        z = output[:, -1]
        expand_z = z.repeat(1, x.shape[1]).view(output.shape)
        decoder_output, _ = self.decoder.forward(expand_z)
        if self.classify:
            net_output = decoder_output.view(decoder_output.shape[1], -1)
            return self.linear(net_output)

        net_output = self.linear(decoder_output)
        return net_output
