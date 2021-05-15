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


def train(train_loader, gradient_clipping=1, hidden_state=10, lr=0.001, opt="adam", epochs=1000):
    model = EncoderDecoder(1, hidden_state, 1)
    if (opt == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    optimizer_name = 'adam' if 'adam' in str(optimizer).lower() else 'mse'

    mse = nn.MSELoss()
    min_loss = float("inf")
    min_in, min_out = None, None
    losses = []
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
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
        losses.append(epoch_loss)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')

        if epoch % 100 == 0:
            path = f'saved_models\\ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt'

            torch.save(model, path)

    path = f'graphs\\ae_toy_{optimizer_name}_lr={lr}_hidden_size={hidden_state_size}_epochs={epochs}_gradient_clipping={gradient_clipping}_'
    _, axis1 = plt.subplots(1, 1)
    axis1.plot(np.arange(0, 50, 1), min_in[0, :, :].detach().numpy())
    axis1.plot(np.arange(0, 50, 1), min_out[0, :, :].detach().numpy())
    axis1.set_xlabel("time")
    # axis1[1].set_xlabel("time")
    axis1.set_ylabel("signal value")
    axis1.legend(("original", "reconstructed"))
    axis1.set_title("time signal reconstruction Example 1 ")
    plt.savefig(path + "example1.jpg")

    _, axis2 = plt.subplots(1, 1)
    axis2.plot(np.arange(0, 50, 1), min_in[1, :, :].detach().numpy())
    axis2.plot(np.arange(0, 50, 1), min_out[1, :, :].detach().numpy())
    axis2.set_xlabel("time")
    # axis1[1].set_xlabel("time")
    axis2.set_ylabel("signal value")
    axis2.legend(("original", "reconstructed"))
    axis2.set_title("time signal reconstruction Example 2 ")
    plt.savefig(path + "example2.jpg")


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


def generate_toy_data(to_int=False):
    random.seed(0)
    if to_int:
        data = torch.randint(0, 10, (10000, 50, 1)).float()
    else:
        data = torch.randn(10000, 50, 1)
    train = data[:6000, :, :]
    validate = data[6000:8000, :, :]
    test = data[8000:, :, :]

    return train, validate, test


if __name__ == '__main__':
    X_train, X_validate, X_test = generate_toy_data()
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.3081,))
    ])
    batch_sizes = [32, 64, 128]

    for batch_size in batch_sizes:

        train_kwargs = {'batch_size': batch_size}
        train_loader = torch.utils.data.DataLoader(X_train, **train_kwargs)
        validate_loader = torch.utils.data.DataLoader(X_validate, **train_kwargs)
        validate_test = torch.utils.data.DataLoader(X_test, **train_kwargs)

        lrs = [0.001, 0.005, 0.01]
        gradient_clip = [1, 2, 0]
        hidden_state_size = [5, 10, 20]
        for lr in lrs:
            for clip in gradient_clip:
                for hidden_state in hidden_state_size:
                    train(train_loader, gradient_clipping=clip, hidden_state=hidden_state,
                          lr=lr)
