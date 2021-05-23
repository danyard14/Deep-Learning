import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, labels_num, classify=False):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.classify = classify
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.reconstruction_linear = nn.Linear(self.hidden_size, output_size)
        self.reconstruct_loss = nn.MSELoss()
        if classify:
            self.classify_linear = nn.Linear(self.hidden_size, labels_num)
            self.cross_entropy = nn.CrossEntropyLoss()

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
        reconstructed_x = self.reconstruction_linear(decoder_output)
        if self.classify:
            decoder_z = decoder_output[:, -1]
            preds_probabilities = self.classify_linear(decoder_z)
            return reconstructed_x, preds_probabilities

        return reconstructed_x

    def loss(self, X, reconstructed_x, target=None, pred_probabilities=None):
        if self.classify:
            cross_entropy_loss = self.cross_entropy(pred_probabilities, target)
            return (self.reconstruct_loss(X, reconstructed_x) + cross_entropy_loss) / 2
        else:
            return self.reconstruct_loss(X, reconstructed_x)
