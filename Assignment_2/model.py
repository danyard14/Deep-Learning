import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, labels_num, is_prediction=False, is_snp=False):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prediction = is_prediction
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.reconstruction_linear = nn.Linear(self.hidden_size, output_size)
        self.mse_loss_layer = nn.MSELoss()
        self.snp = is_snp
        if is_prediction:
            self.prediction_linear = nn.Linear(self.hidden_size, labels_num)
            self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: input of shape (batch X sequence_size X num_of_features)
        :return:
        """
        output, _ = self.encoder.forward(x)
        z = output[:, -1]
        expand_z = z.repeat(1, x.shape[1]).view(output.shape)
        decoder_output, _ = self.decoder.forward(expand_z)
        reconstructed_x = self.reconstruction_linear(decoder_output)
        if self.prediction:
            decoder_z = decoder_output if self.snp else decoder_output[:, -1]
            preds = self.prediction_linear(decoder_z)
            return reconstructed_x, preds

        return reconstructed_x

    def loss(self, X, reconstructed_x, target=None, preds=None):
        if self.prediction and self.snp:
            return (self.mse_loss_layer(X, reconstructed_x) + self.mse_loss_layer(preds, target)) / 2
        elif self.prediction:
            return (self.mse_loss_layer(X, reconstructed_x) + self.cross_entropy(preds, target)) / 2
        else:
            return self.mse_loss_layer(X, reconstructed_x)
