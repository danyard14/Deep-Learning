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


def train(train_loader, validate_data, device, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam", epochs=1000,
          batch_size=32):
    model = EncoderDecoder(1, hidden_state_size, 1, 50).to(device)
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
    for epoch in range(0, epochs):
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

        if epoch % 100 == 0:
            path = f'{results_path}saved_models/ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}_' \
                   f'_gradient_clipping={gradient_clipping}_'
            create_folders(path)
            torch.save(model, path + f"/epoch={epoch}_bestloss={best_loss_global}.pt")

            # run validation        if epoch % 20 == 0:
            model.eval()
            mse.eval()
            output = model(validate_data)
            loss = mse(output, validate_data)  # print("Accuracy: {:.4f}".format(acc))
            validation_losses.append(loss.item())
            mse.train()
            model.train()

    plot_sequence_examples(epochs, gradient_clipping, lr, min_in, min_out, optimizer_name, batch_size)

    plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, batch_size)


def plot_validation_loss(epochs, gradient_clipping, lr, optimizer_name, validation_losses, batch_size):
    path = f'{results_path}graphs/validation/ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}_epochs={epochs}' \
           f'_gradient_clipping={gradient_clipping}_batch_size={batch_size}_'
    create_folders(path)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(1, len(validation_losses) + 1, 1), validation_losses)
    axis1.set_xlabel("epochs X 50")
    axis1.set_ylabel("validation loss")
    axis1.set_title("validation loss")
    plt.savefig(path + "/loss.jpg")


def plot_sequence_examples(epochs, gradient_clipping, lr, min_in, min_out, optimizer_name, batch_size):
    path = f'{results_path}graphs/sequence_examples/ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}' \
           f'_epochs={epochs}_gradient_clipping={gradient_clipping}_batch_size={batch_size}_'
    create_folders(path)
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(0, 50, 1), min_in[0, :, :].detach().cpu().numpy())
    axis1.plot(np.arange(0, 50, 1), min_out[0, :, :].detach().cpu().numpy())
    axis1.set_xlabel("time")
    # axis1[1].set_xlabel("time")
    axis1.set_ylabel("signal value")
    axis1.legend(("original", "reconstructed"))
    axis1.set_title("time signal reconstruction Example 1 ")
    plt.savefig(path + "/example1.jpg")
    _, axis2 = plt.subplots(1, 1)
    axis2.plot(np.arange(0, 50, 1), min_in[1, :, :].detach().cpu().numpy())
    axis2.plot(np.arange(0, 50, 1), min_out[1, :, :].detach().cpu().numpy())
    axis2.set_xlabel("time")
    # axis1[1].set_xlabel("time")
    axis2.set_ylabel("signal value")
    axis2.legend(("original", "reconstructed"))
    axis2.set_title("time signal reconstruction Example 2 ")
    plt.savefig(path + "/example2.jpg")


def validate(model, device, test_loader):
    # model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    results_path = "/home/mosesofe/results/pdl_Ass2/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_validate, X_test = generate_toy_data()
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.3081,))
    ])
    batch_sizes = [40, 80, 10, 5]

    for batch_size in batch_sizes:

        train_kwargs = {'batch_size': batch_size}
        train_loader = torch.utils.data.DataLoader(X_train, **train_kwargs)
        validate_loader = torch.utils.data.DataLoader(X_validate, **train_kwargs)
        validate_test = torch.utils.data.DataLoader(X_test, **train_kwargs)

        lrs = [0.01, 0.001, 0.005]
        gradient_clip = [0, 1, 2]
        hidden_state_sizes = [120, 60, 30, 10]
        for lr in lrs:
            for clip in gradient_clip:
                for hidden_state_size in hidden_state_sizes:
                    train(train_loader, X_validate, device, gradient_clipping=clip, hidden_state_size=hidden_state_size,
                          lr=lr, batch_size=batch_size)
