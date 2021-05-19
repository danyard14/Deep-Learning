from torchvision.transforms import Lambda

from model import EncoderDecoder
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import EncoderDecoder


def train(train_loader, test_loader, batch_size, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam",
          epochs=10,
          classify=True):
    model = EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=1, T=784) if not classify \
        else EncoderDecoder(input_size=1, hidden_size=hidden_state_size, output_size=10, T=784, classify=True)
    model = model.to(device)
    loss_layer = nn.MSELoss() if not classify else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_name = "mse" if not classify else "cross_entropy"
    min_loss = float("inf")
    min_in, min_out = None, None
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            data_sequential = data.view(data.shape[0], 784, 1)
            optimizer.zero_grad()
            output = model(data_sequential)
            if classify:
                loss = loss_layer(output, target)
            else:  # reconstruction
                loss = loss_layer(output, data_sequential)
            total_loss += loss.item()
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

            optimizer.step()

        epoch_loss = total_loss / len(train_loader)
        print(f'Train Epoch: {epoch} \t loss: {epoch_loss}')
        min_loss = min(epoch_loss, min_loss)

        if epoch % 100 == 0:
            path = f'saved_models\\ae_toy_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt'
            torch.save(model, path)

        # run test
        if classify:
            total = 0
            correct = 0
            model.eval()
            for batch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                data_sequential = data.view(data.shape[0], 784, 1)
                output = model(data_sequential)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy : {(100 * correct / total)}')
            model.train()
    #
    # plot_sequence_examples(epochs, gradient_clipping, lr, min_in, min_out, loss_name, batch_size)
    #
    # plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, batch_size)


# def validate(model, device, test_loader):
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


def divide_by_255(x):
    x /= 255
    return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(divide_by_255),
        transforms.Normalize((0.5,), (0.3081,))
    ])
    batch_size = 64
    train_kwargs = {'batch_size': batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                               transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **train_kwargs)
    classify = False

    # lrs = []
    # gradient_clip = []
    # hidden_state_size = []
    # for lr in lrs:
    #     for clip in gradient_clip:
    #         for hidden_state in hidden_state_size:

    train(train_loader, test_loader, batch_size, gradient_clipping=1, hidden_state_size=300, lr=0.001, opt="adam",
          epochs=10)
