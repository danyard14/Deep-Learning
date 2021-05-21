import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import Lambda

from model import EncoderDecoder
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import EncoderDecoder
import os.path
import pandas as pd


def train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam",
          epochs=15,
          classify=True):
    model = EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1, T=784) if not classify \
        else EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=10, T=784, classify=True)
    model = model.to(device)
    loss_layer = nn.MSELoss() if not classify else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_name = "mse" if not classify else "cross_entropy"
    min_loss = float("inf")
    task_name = "classify" if classify else "reconstruct"
    validation_losses = []
    validation_accuracies = []
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            data_sequential = data.view(data.shape[0], 784, 1)  # turn each image to vector sized 784
            target = target if classify else data_sequential
            optimizer.zero_grad()
            output = model(data_sequential)
            loss = loss_layer(output, target)
            total_loss += loss.item()
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
            optimizer.step()

        epoch_loss = total_loss / len(train_loader)

        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')

        validation(model, loss_layer, test_loader, validation_losses, device, classify,
                   validation_accuracies)

        if epoch % 1 == 0 or epoch_loss < min_loss:
            file_name = f"ae_toy_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt"
            path = os.path.join("saved_models", "MNIST_task", task_name, file_name)
            torch.save(model, path)

        min_loss = min(epoch_loss, min_loss)

    plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, hidden_state_size, task_name)
    if classify:
        plot_validation_acc(epochs, gradient_clipping, lr, loss_name, validation_accuracies, hidden_state_size,
                            task_name)


def validation(model, loss_layer, test_loader, validation_losses, device, classification, validation_accuracies):
    total_loss = 0
    total_samples = 0 if classification else 1  # else to avoid div by 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_sequential = data.view(data.shape[0], 784, 1)
            target = target if classification else data_sequential
            output = model(data_sequential)
            total_loss += loss_layer(output, target)  # print("Accuracy: {:.4f}".format(acc))

            if classification:
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = total_loss / len(test_loader)
        epoch_acc = (100 * correct / total_samples)
        validation_losses.append(epoch_loss)
        validation_accuracies.append(epoch_acc)
        print(f"validation loss = {epoch_loss}, validation acc = {epoch_acc}")


def plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, hidden_state, task_name):
    file_name = f'ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state}_epochs={epochs}' \
                f'_gradient_clipping={gradient_clipping}'
    path = os.path.join("graphs", "MNIST_task", task_name, file_name)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(validation_losses) + 1, 1), validation_losses)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel("validation loss")
    axis1.set_title("validation loss")
    plt.savefig(path + "loss.jpg")


def plot_validation_acc(epochs, gradient_clipping, lr, optimizer_name, validation_accs, hidden_state_size, task_name):
    file_name = f'ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}_epochs={epochs}' \
                f'_gradient_clipping={gradient_clipping}'
    path = os.path.join("graphs", "MNIST_task", task_name, file_name)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(validation_accs) + 1, 1), validation_accs)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel("accuracy")
    axis1.set_title("validation accuracy")
    plt.savefig(path + "acc.jpg")


# def validate_classification(model, loss_layer, test_loader, validation_losses, device, classification):  # TODO: delete
#     validation()
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

class Normalizer:
    def __init__(self):
        self.max_stock_values = None
        self.min_stock_values = None
        self.mean_stock_values = None

    def normalize(self, stock_sequences):
        """
           :param stock_sequences = np.array of shape (num of sequences X sequence length)
           normalizes each sequence to range of [0,1] with mean=0.5
        """
        self.max_stock_values = stock_sequences.max(axis=1).reshape(-1, 1)
        self.min_stock_values = stock_sequences.min(axis=1).reshape(-1, 1)
        stock_sequences = (stock_sequences - self.min_stock_values) / (self.max_stock_values - self.min_stock_values)
        self.mean_stock_values = stock_sequences.mean(axis=1).reshape(-1, 1)
        normalized_stock_sequence = stock_sequences / (2 * self.mean_stock_values)

        return normalized_stock_sequence

    def undo_normalize(self, stock_sequences):
        assert self.mean_stock_values is not None, "must normalize before undo normalize"
        stock_sequences = stock_sequences * (2 * self.mean_stock_values)
        stock_sequences_unnormalized = (stock_sequences * (
                self.max_stock_values - self.min_stock_values)) + self.min_stock_values
        return stock_sequences_unnormalized


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.3081, 0.3081, 3081))
        # TODO: check if this transform meets demands of scale (0,1) and mean=0.5
    ])

    stocks_data = pd.read_csv(os.path.join('..', 'data', 'SP 500 Stock Prices 2014-2017.csv'))
    amazon_stocks = stocks_data[stocks_data['symbol'] == 'AMZN']["high"].values
    google_stocks = stocks_data[stocks_data['symbol'] == 'GOOGL']["high"].values
    # TODO: implement and call plot high stocks of google&amazon
    high_stocks = stocks_data["high"].values
    stock_names = stocks_data['symbol'].unique()
    stock_sequences = np.zeros((len(stock_names) - 26, *amazon_stocks.shape))
    filtered_stocks_names = []
    i = 0
    for stock in stock_names:
        temp = stocks_data[stocks_data['symbol'] == stock]["high"].values
        if temp.shape[0] == stock_sequences.shape[1]:
            stock_sequences[i, :] = temp.copy()
            filtered_stocks_names.append(stock)
            i += 1
    normalizer = Normalizer()
    normalized_tensor_sequences = torch.tensor(normalizer.normalize(stock_sequences)).to(device)

    train_loader = torch.utils.data.DataLoader(amazon_stocks.values, transforms=transform)
    test_loader = torch.utils.data.DataLoader(google_stocks.values, transforms=transform)
    classify = False

    hidden_state_sizes = [64]
    lrs = [0.001]
    gradient_clip = [1, 0]
    for lr in lrs:
        for clip in gradient_clip:
            for hidden_state_size in hidden_state_sizes:
                train(train_loader, test_loader, gradient_clipping=clip, hidden_state_size=hidden_state_size, lr=lr,
                      opt="adam")
