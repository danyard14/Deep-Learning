import numpy as np
from matplotlib import pyplot as plt
from Assignment_2.data_sets import SnP500_dataset
import torch
import torch.nn as nn
from model import EncoderDecoder
import os.path
import pandas as pd
from tensorboardX import SummaryWriter
from pathlib import Path


def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def init_writer(lr, classify, hidden_size, epochs, task_name):
    path = os.path.join(results_path, "tensorboard", "s&p500",
                        f"_AE_S&P500_classify={classify}_lr={lr}_hidden_size={hidden_size}_epochs={epochs}_{task_name}")
    create_folders(path)
    writer = SummaryWriter(logdir=path)
    return writer


def train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=10, lr=0.001,
          epochs=3000, is_prediction=False):
    model = EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1,
                           labels_num=1) if not is_prediction \
        else EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1, is_prediction=True,
                            labels_num=1, is_snp=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_name = "mse"
    min_loss = float("inf")
    task_name = "classify" if is_prediction else "reconstruct"
    validation_losses = []
    tensorboard_writer = init_writer(lr, is_prediction, hidden_state_size, epochs, task_name)
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data_sequential = (data.view(data.shape[0], data.shape[1], 1)).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            if is_prediction:
                resconstucted_batch, batch_preds = model(data_sequential)
                batch_preds = batch_preds.view(batch_preds.shape[0], batch_preds.shape[1])
                loss = model.loss(data_sequential, resconstucted_batch, target, batch_preds)
            else:
                resconstucted_batch = model(data_sequential)
                loss = model.loss(data_sequential, resconstucted_batch)
            total_loss += loss.item()
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        tensorboard_writer.add_scalar('train_loss', epoch_loss, epoch)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')

        validation_loss = validation(model, test_loader, validation_losses, device, is_prediction,
                                     tensorboard_writer, epoch)

        if epoch % 5 == 0 or validation_loss < min_loss:
            file_name = f"ae_s&p500_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt"
            path = os.path.join(results_path, "saved_models", "s&p500_task", task_name, file_name)
            torch.save(model, path)

        min_loss = min(validation_loss, min_loss)

    plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, hidden_state_size, task_name)


def validation(model, test_loader, validation_losses, device, is_prediction, tensorboard_writer, epoch):
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_sequential = (data.view(data.shape[0], data.shape[1], 1)).to(device)
            if is_prediction:
                resconstucted_batch, batch_preds = model(data_sequential)
                batch_preds = batch_preds.view(batch_preds.shape[0], batch_preds.shape[1])
                loss = model.loss(data_sequential, resconstucted_batch, target, batch_preds)
            else:
                resconstucted_batch = model(data_sequential)
                loss = model.loss(data_sequential, resconstucted_batch)
            total_loss += loss.item()  # print("Accuracy: {:.4f}".format(acc))

        epoch_loss = total_loss / len(test_loader)
        tensorboard_writer.add_scalar('validation_loss', epoch_loss, epoch)
        validation_losses.append(epoch_loss)
        print(f"validation loss = {epoch_loss}")

        return epoch_loss


def plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, hidden_state, task_name):
    file_name = f'ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state}_epochs={epochs}' \
                f'_gradient_clipping={gradient_clipping}'
    path = os.path.join(results_path, "graphs", "s&p500_task", task_name, file_name)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(validation_losses) + 1, 1), validation_losses)
    axis1.set_xlabel("epochs")
    axis1.set_ylabel("validation loss")
    axis1.set_title("validation loss")
    plt.savefig(path + "loss.jpg")


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
        """
        :param stock_sequences: batch of stocks sized (batch_size X sequence_length)
        """
        batch_size, T = stock_sequences.shape
        assert self.mean_stock_values is not None, "must normalize before undo normalize"
        stock_sequences = stock_sequences * (2 * self.mean_stock_values[:batch_size, :])
        stock_sequences_unnormalized = (stock_sequences * (
                self.max_stock_values[:batch_size, :] - self.min_stock_values[:batch_size, :])) + self.min_stock_values[
                                                                                                  :batch_size, :]
        return stock_sequences_unnormalized


def plot_amazon_google_high_stocks(stocks_df):
    stocks_df["date"] = pd.to_datetime(stocks_df.date)

    amazon_stocks = stocks_data[stocks_data['symbol'] == 'AMZN'][["date", "high"]]

    google_stocks = stocks_data[stocks_data['symbol'] == 'GOOGL'][["date", "high"]]

    _, axis1 = plt.subplots(1, 1)
    axis1.plot(google_stocks['date'], google_stocks['high'].values)
    axis1.plot(amazon_stocks['date'], amazon_stocks['high'].values)
    plt.xticks(rotation=30)
    plt.title("amazon and google max stock values, years 2017-2017")
    plt.legend(("google", "amazon"))

    plt.show()
    return google_stocks['date']


def plot_reconstructed_x(data, reconsturct, i, dates, stock_names):
    _, axis = plt.subplots(1, 1)
    _, T = data.shape
    axis.plot(dates[:-1], data[i, :])
    axis.plot(dates[:-1], reconsturct[i, :])
    plt.xticks(rotation=30)
    axis.set_title(f"{stock_names[i]} Stock reconstruction example")
    axis.legend(("original", "reconstructed"))
    axis.set_xlabel("Time")
    axis.set_ylabel("Max Stock Value")
    file_name = f'ae_snp_sequence_reconstructed_plots_{i + 1}'
    path = os.path.join(results_path, "graphs", "s&p500_task", "classify_model", file_name)
    create_folders(path)

    plt.savefig(path)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    results_path = ""
    # results_path = os.path.join("/home", "mosesofe", "results", "pdl_Ass2")  # for offline runs

    stocks_data = pd.read_csv(os.path.join('..', 'data', 'SP 500 Stock Prices 2014-2017.csv'))
    dates = plot_amazon_google_high_stocks(stocks_data)
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

    normalizer_train = Normalizer()
    normalizer_validation = Normalizer()
    normalizer_test = Normalizer()

    num_of_stocks = stock_sequences.shape[0]
    normalized_train = normalizer_train.normalize(stock_sequences[:int(num_of_stocks * 0.6), :])
    normalized_validation = normalizer_validation.normalize(
        stock_sequences[int(num_of_stocks * 0.6):int(num_of_stocks * 0.8), :])
    normalized_test = normalizer_test.normalize(stock_sequences[int(num_of_stocks * 0.8):, :])

    examples_train = normalized_train[:, :-1].copy()
    examples_validation = normalized_validation[:, :-1].copy()
    examples_test = normalized_test[:, :-1].copy()
    targets_train = normalized_train[:, 1:].copy()
    targets_validation = normalized_validation[:, 1:].copy()
    targets_test = normalized_test[:, 1:].copy()

    train_X = torch.tensor(examples_train, dtype=torch.float32)
    validation_X = torch.tensor(examples_validation, dtype=torch.float32)
    test_X = torch.tensor(examples_test, dtype=torch.float32)

    train_Y = torch.tensor(targets_train, dtype=torch.float32)
    validation_Y = torch.tensor(targets_validation, dtype=torch.float32)
    test_Y = torch.tensor(targets_test, dtype=torch.float32)

    train_data = SnP500_dataset(train_X, train_Y)
    validation_data = SnP500_dataset(validation_X, validation_Y)
    test_data = SnP500_dataset(test_X, test_Y)
    test_stock_names = filtered_stocks_names[int(num_of_stocks * 0.8):]

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=4)
    validation_loader = torch.utils.data.DataLoader(validation_data, shuffle=True, batch_size=4)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=len(test_stock_names))


    # Grid search
    hidden_state_sizes = [130, 180, 240, 300]
    lrs = [0.001]
    gradient_clip = [1]
    for lr in lrs:
        for clip in gradient_clip:
            for hidden_state_size in hidden_state_sizes:
                train(train_loader, validation_loader, gradient_clipping=clip, hidden_state_size=hidden_state_size,
                      lr=lr, is_prediction=False)

    # use best model to plot original vs reconstructed digit images
    model = torch.load(
        r"C:\Users\t-ofermoses\PycharmProjects\pdl\Assignment_2\saved_models\s&p500_task\reconstruct\best\ae_s&p500_mse_lr=0.001_hidden_size=130_epoch=2990_gradient_clipping=1.pt")

    model = model.to(device)
    with torch.no_grad():
        for data, targets in test_loader:
            data_sequential = (data.view(data.shape[0], data.shape[1], 1)).to(device)
            reconstruct, preds = model(data_sequential)
            # reconstruct = reconstruct.cpu().numpy().reshape(
            #     (data_sequential.shape[0], data_sequential.shape[1]))
            preds = preds.cpu().numpy().reshape(
                (targets.shape[0], targets.shape[1]))
            # denormalized_Xs = normalizer_test.undo_normalize(
            #     data_sequential.cpu().numpy().reshape((data_sequential.shape[0], data_sequential.shape[1])))
            denorm_targets = normalizer_test.undo_normalize(
                targets.cpu().numpy().reshape(targets.shape[0], targets.shape[1]))
            # denormalized_reconstructed_Xs = normalizer_test.undo_normalize(reconstruct)
            denorm_preds = normalizer_test.undo_normalize(preds)
            for i in range(len(test_stock_names) // 3):
                plot_reconstructed_x(denorm_preds, denorm_targets, i, dates, test_stock_names)
            break
