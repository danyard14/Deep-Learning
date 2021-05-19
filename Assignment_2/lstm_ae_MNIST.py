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


def train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam",
          epochs=10,
          classify=False):
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
        min_loss = min(epoch_loss, min_loss)

        validation(model, loss_layer, test_loader, validation_losses)  # calculate

        if epoch % 10 == 0:
            file_name = f"ae_toy_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt"
            path = os.path.join("saved_models", "MNIST_task", task_name, file_name)
            torch.save(model, path)

    #
    # plot_sequence_examples(epochs, gradient_clipping, lr, min_in, min_out, loss_name, batch_size)
    #
    # plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, batch_size)


def validation(model, loss_layer, test_loader, validation_losses, device, classification, validation_accuracies):
    total_loss = 0
    total_acc = 0
    total_samples = 0 if classification else 1  # else to avoid div by 0
    correct = 0
    with torch.no_grad:
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


def validate_classification(model, loss_layer, test_loader, validation_losses, device, classification):  # TODO: delete
    validation()
    model.eval()
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.3081,))
        # TODO: check if this transform meets demands of scale (0,1) and mean=0.5
    ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                               transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=20)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=20)
    classify = False

    # lrs = []
    # gradient_clip = []
    # hidden_state_size = []
    # for lr in lrs:
    #     for clip in gradient_clip:
    #         for hidden_state in hidden_state_size:

    train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=64, lr=0.001, opt="adam",
          epochs=10)
