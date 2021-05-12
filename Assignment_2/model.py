import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.LSTM(self.input_size, self.hidden_size)
        self.decoder = nn.LSTM(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)  # TODO: check if needed

    def forward(self, x):
        output, (h_n, c_n) = self.encoder.forward(x)
        output, _ = self.decoder.forward(h_n)
        return self.linear(output)
