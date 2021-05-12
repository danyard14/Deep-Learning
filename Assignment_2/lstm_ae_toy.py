import random

from model import EncoderDecoder
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch AE Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--dataset', default=10, type=int, help='modelnet 10 or 40')
parser.add_argument('--opt', default="adam", type=str, help='choose optimizer')
parser.add_argument('--hidden_size', default=64, type=int, help="")


def train(args, model, train_loader, optimizer, gradient_clipping=True):
    # model.X_train()
    mse = nn.MSELoss()
    losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = mse(output, data)
            total_loss += loss.item()
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')


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


def generate_toy_data(to_int=True):
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
    args = parser.parse_args()
    X_train, X_validate, X_test = generate_toy_data()
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.3081,))
    ])
    train_kwargs = {'batch_size': args.batch_size}
    train_loader = torch.utils.data.DataLoader(X_train, **train_kwargs)
    validate_loader = torch.utils.data.DataLoader(X_validate, **train_kwargs)
    validate_test = torch.utils.data.DataLoader(X_test, **train_kwargs)
    model = EncoderDecoder(1, args.hidden_size, 1)

    if (args.opt == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(args, model, train_loader, optimizer)
