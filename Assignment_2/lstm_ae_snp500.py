import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import Lambda

from Assignment_2.data_sets import SnP500_dataset
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
from tensorboardX import SummaryWriter


def init_writer(lr, classify, hidden_size, epochs):
    path = os.path.join("tensorboard", "s&p500")
    writer = SummaryWriter(logdir=path,
                           comment=f"_AE_S&P500_classify={classify}_lr={lr}_hidden_size={hidden_size}_epochs={epochs}")
    return writer


# TODO: remove opt from params (in all scripts)
def train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam",
          epochs=300,
          classify=False):
    model = EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1, labels_num=10) if not classify \
        else EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1, classify=True, labels_num=10)
    model = model.to(device)
    loss_layer = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_name = "mse"
    min_loss = float("inf")
    task_name = "classify" if classify else "reconstruct"
    validation_losses = []
    validation_accuracies = []
    tensorboard_writer = init_writer(lr, classify, hidden_state_size, epochs)
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data_sequential = (data.view(data.shape[0], data.shape[1], 1)).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            if classify:
                resconstucted_batch, batch_pred_probs = model(data_sequential)
                loss = (loss_layer(data_sequential, resconstucted_batch) + model.cross_entropy(batch_pred_probs,
                                                                                               target)) / 2
            else:
                resconstucted_batch = model(data_sequential)
                if len(torch.isnan(resconstucted_batch).unique()) > 1:
                    print("found nan")
                loss = loss_layer(data_sequential, resconstucted_batch)
            total_loss += loss.item()
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        tensorboard_writer.add_scalar('train_loss', epoch_loss, epoch)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')

        validation_loss = validation(model, loss_layer, test_loader, validation_losses, device, classify,
                                     validation_accuracies, tensorboard_writer, epoch)

        if epoch % 5 == 0 or validation_loss < min_loss:
            file_name = f"ae_s&p500_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt"
            path = os.path.join("saved_models", "s&p500_task", task_name, file_name)
            torch.save(model, path)

        min_loss = min(validation_loss, min_loss)

    plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, hidden_state_size, task_name)
    if classify:
        plot_validation_acc(epochs, gradient_clipping, lr, loss_name, validation_accuracies, hidden_state_size,
                            task_name)


def validation(model, loss_layer, test_loader, validation_losses, device, classification, validation_accuracies
               , tensorboard_writer, epoch):
    total_loss = 0
    total_samples = 0 if classification else 1  # else to avoid div by 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_sequential = (data.view(data.shape[0], data.shape[1], 1)).to(device)
            output = model(data_sequential)
            if len(torch.isnan(output).unique()) > 1:
                print("found nan")
            total_loss += loss_layer(output, data_sequential)  # print("Accuracy: {:.4f}".format(acc))

            if classification:
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = total_loss / len(test_loader)
        epoch_acc = (100 * correct / total_samples)
        tensorboard_writer.add_scalar('validation_loss', epoch_loss, epoch)
        tensorboard_writer.add_scalar('validation_acc', epoch_acc, epoch)
        validation_losses.append(epoch_loss)
        validation_accuracies.append(epoch_acc)
        print(f"validation loss = {epoch_loss}, validation acc = {epoch_acc}")

        return epoch_loss


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


def plot_amazon_google_high_stocks(stocks_df):
    stocks_df["date"] = pd.to_datetime(stocks_df.date)

    amazon_stocks = stocks_data[stocks_data['symbol'] == 'AMZN'][["date", "high"]]
    # dates = [pd.datetime.datetime.strptime(d, "%m/%d/%Y").date() for d in amazon_stocks['dates'].values]

    google_stocks = stocks_data[stocks_data['symbol'] == 'GOOGL'][["date", "high"]]

    _, axis1 = plt.subplots(1, 1)
    axis1.plot(google_stocks['date'], google_stocks['high'].values)
    axis1.plot(amazon_stocks['date'], amazon_stocks['high'].values)
    plt.xticks(rotation=45)
    plt.title("amazon and google max stock values, years 2017-2017")
    plt.legend(("google","amazon"))

    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    results_path = ""
    # results_path = os.path.join("/home", "mosesofe", "results", "pdl_Ass2") #for offline runs

    stocks_data = pd.read_csv(os.path.join('..', 'data', 'SP 500 Stock Prices 2014-2017.csv'))
    plot_amazon_google_high_stocks(stocks_data)
    # TODO: implement and call plot high stocks of google&amazon
    high_stocks = stocks_data["high"].values
    stock_names = stocks_data['symbol'].unique()
    stock_sequences = np.zeros((len(stock_names) - 28, 1007))
    filtered_stocks_names = []
    i = 0
    for stock in stock_names:
        temp = stocks_data[stocks_data['symbol'] == stock]["high"].values
        if temp.shape[0] == stock_sequences.shape[1] and np.isnan(temp).sum() == 0:
            stock_sequences[i, :] = temp.copy()
            filtered_stocks_names.append(stock)
            i += 1

    normalizer = Normalizer()
    np.random.shuffle(stock_sequences)
    normalized_data = normalizer.normalize(stock_sequences)
    examples = normalized_data[:, :-1].copy()
    targets = normalized_data[:, 1:].copy()
    num_of_stocks = normalized_data.shape[0]
    train_X = torch.tensor(examples[0:int(num_of_stocks * 0.6), :], dtype=torch.float32)
    validation_X = torch.tensor(examples[int(num_of_stocks * 0.6):int(num_of_stocks * 0.8), :], dtype=torch.float32)
    test_X = torch.tensor(examples[int(num_of_stocks * 0.8):, :], dtype=torch.float32)
    train_Y = torch.tensor(targets[0:int(num_of_stocks * 0.6), :], dtype=torch.float32)
    validation_Y = torch.tensor(targets[int(num_of_stocks * 0.6):int(num_of_stocks * 0.8), :], dtype=torch.float32)
    test_Y = torch.tensor(targets[int(num_of_stocks * 0.8):, :], dtype=torch.float32)
    train_data = SnP500_dataset(train_X, train_Y)
    validation_data = SnP500_dataset(validation_X, validation_Y)
    test_data = SnP500_dataset(test_X, test_Y)

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=4)
    validation_loader = torch.utils.data.DataLoader(validation_data, shuffle=True, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=4)

    hidden_state_sizes = [64, 100, 150, 200]
    lrs = [0.001]
    gradient_clip = [1, 0]
    for lr in lrs:
        for clip in gradient_clip:
            for hidden_state_size in hidden_state_sizes:
                train(train_loader, validation_loader, gradient_clipping=clip, hidden_state_size=hidden_state_size,
                      lr=lr,
                      opt="adam")
