import os.path
import random

import numpy as np

from model import EncoderDecoder
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path


def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def train(train_loader, validate_data, device, gradient_clipping=1, hidden_state=10, lr=0.001, opt="adam", epochs=600,
          batch_size=32):
    model = EncoderDecoder(1, hidden_state, 1, 50).to(device)
    validate_data = validate_data.to(device)
    if (opt == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    optimizer_name = 'adam' if 'adam' in str(optimizer).lower() else 'mse'

    mse = nn.MSELoss()
    min_loss = float("inf")
    best_loss_global = float("inf")
    min_in, min_out = None, None
    validation_losses = []
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = mse(output, data)
            total_loss += loss.item()
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_in, min_out = data, output
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        best_loss_global = min(best_loss_global, epoch_loss)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')

        if epoch % 50 == 0:
            file_name = f'ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state}_' \
                        f'_gradient_clipping={gradient_clipping}'
            path = os.path.join("saved_models", "toy_task", file_name)
            create_folders(path)
            torch.save(model, os.path.join(path, f'epoch={epoch}_bestloss={best_loss_global}.pt'))
        if epoch % 10 == 0:
            validation(model, mse, validate_data, validation_losses)

    plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, batch_size, hidden_state)


def validation(model, mse, validate_data, validation_losses):
    model.eval()
    mse.eval()
    output = model(validate_data)
    loss = mse(output, validate_data)  # print("Accuracy: {:.4f}".format(acc))
    print(f"validation loss = {loss}")
    validation_losses.append(loss.item())
    mse.train()
    model.train()


def plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, batch_size, hidden_state):
    file_name = f'ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state}_epochs={epochs}' \
                f'_gradient_clipping={gradient_clipping}_batch_size={batch_size}_'
    path = os.path.join("graphs", "toy_task", "validation", file_name)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(validation_losses) + 1, 1), validation_losses)
    axis1.set_xlabel("epochs X 50")
    axis1.set_ylabel("validation loss")
    axis1.set_title("validation loss")
    plt.savefig(path + "loss.jpg")


def plot_sequence_examples(original_xs, reconstructed_xs, path):
    save_path = os.path.join("graphs", "toy_task", "sequence_examples", path)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(0, 50, 1), original_xs[0, :, :].detach().cpu().numpy())
    # axis1.plot(np.arange(0, 50, 1), reconstructed_xs[0, :, :].detach().cpu().numpy())
    axis1.set_xlabel("time")
    axis1.set_ylabel("signal value")
    axis1.legend(("original", "reconstructed"))
    axis1.set_title("time signal reconstruction Example 1 ")
    plt.savefig(save_path + "example1.jpg")
    _, axis2 = plt.subplots(1, 1)
    axis2.plot(np.arange(0, 50, 1), original_xs[1, :, :].detach().cpu().numpy())
    axis2.plot(np.arange(0, 50, 1), reconstructed_xs[1, :, :].detach().cpu().numpy())
    axis2.set_xlabel("time")
    axis2.set_ylabel("signal value")
    axis2.legend(("original", "reconstructed"))
    axis2.set_title("time signal reconstruction Example 2 ")
    plt.savefig(save_path + "example2.jpg")

def plot_sequence_examples(original_xs, path):
    save_path = os.path.join("graphs", "toy_task", "sequence_examples", path)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(0, 50, 1), original_xs[8, :, :].detach().cpu().numpy())
    axis1.set_xlabel("time")
    axis1.set_ylabel("signal value")
    axis1.set_title("signal Example 1 ")
    plt.savefig(save_path + "example1.jpg")
    _, axis2 = plt.subplots(1, 1)
    axis2.plot(np.arange(0, 50, 1), original_xs[9, :, :].detach().cpu().numpy())
    axis2.set_xlabel("time")
    axis2.set_ylabel("signal value")
    axis2.set_title(" signal  Example 2 ")
    plt.savefig(save_path + "example2.jpg")


def generate_toy_data(n_sequences=10000, T=50, to_int=False):
    random.seed(0)
    if to_int:
        data = torch.randint(0, 10, (10000, 50, 1)).float()
    else:
        data = torch.FloatTensor(n_sequences, T, 1).uniform_(0, 1)

    train_data = data[:6000, :, :]
    validate_data = data[6000:8000, :, :]
    test_data = data[8000:, :, :]

    return train_data, validate_data, test_data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_validate, X_test = generate_toy_data()
    batch_sizes = [40]

    # for batch_size in batch_sizes:
    #
    #     train_kwargs = {'batch_size': batch_size}
    #     train_loader = torch.utils.data.DataLoader(X_train, **train_kwargs, shuffle=True)
    #     validate_loader = torch.utils.data.DataLoader(X_validate, **train_kwargs, shuffle=True)
    #     validate_test = torch.utils.data.DataLoader(X_test, **train_kwargs, shuffle=True)
    #
    #     lrs = [0.001, 0.01]
    #     gradient_clip = [1, 0]
    #     hidden_state_sizes = [120, 60, 30]
    #     for lr in lrs:
    #         for clip in gradient_clip:
    #             for hidden_state in hidden_state_sizes:
    #                 train(train_loader, X_validate, device, gradient_clipping=clip, hidden_state=hidden_state,
    #                       lr=lr, batch_size=batch_size)

    """
    after grid search, best params are:
    lr = 
    gradient clipping = 
    hidden state size = 
    
    running model on test:
    """

    file_name = "ae_toy_adam_lr=0.001_hidden_size=60__gradient_clipping=1"
    model_path = r"C:\Users\t-ofermoses\PycharmProjects\pdl\Assignment_2\saved_models\toy_task\ae_toy_adam_lr=0.001_hidden_size=60__gradient_clipping=1\epoch=550_bestloss=0.0018718887446448208.pt"
    model = torch.load(model_path)
    # # model = EncoderDecoder(1, 54, 1, 50)
    # mse = nn.MSELoss()
    # mse.eval()
    # model.eval()
    # X_test = X_test.to(device)
    # test_output = model(X_test)
    # test_loss = mse(test_output, X_test)
    # print(f"test loss = {test_loss}")
    # plot_sequence_examples(X_test, test_output, file_name)
    plot_sequence_examples(X_test,file_name)
