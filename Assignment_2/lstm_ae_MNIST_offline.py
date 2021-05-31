import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import EncoderDecoder
import os.path
from tensorboardX import SummaryWriter


def init_writer(results_path, lr, classify, hidden_size, epochs):
    path = os.path.join(results_path, "tensorboard", "MNIST")
    writer = SummaryWriter(logdir=path,
                           comment=f"_AE_MNIST_classify={classify}_lr={lr}_hidden_size={hidden_size}_epochs={epochs}")
    return writer


def train(train_loader, test_loader, gradient_clipping=1, hidden_state_size=10, lr=0.001, opt="adam",
          epochs=400,
          classify=True):
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
    tensorboard_writer = init_writer(results_path, lr, classify, hidden_state_size, epochs)
    for epoch in range(1, epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            data_sequential = data.view(data.shape[0], 784, 1)  # turn each image to vector sized 784
            optimizer.zero_grad()
            if classify:
                resconstucted_batch, batch_pred_probs = model(data_sequential)
                loss = (loss_layer(data_sequential, resconstucted_batch) + model.cross_entropy(batch_pred_probs,
                                                                                               target)) / 2
            else:
                resconstucted_batch = model(data_sequential)
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
            file_name = f"ae_toy_{loss_name}_lr={lr}_hidden_size={hidden_state_size}_epoch={epoch}_gradient_clipping={gradient_clipping}.pt"
            path = os.path.join(results_path, "saved_models", "MNIST_task", task_name, file_name)
            torch.save(model, path)

        min_loss = min(validation_loss, min_loss)

    plot_validation_loss(epochs, gradient_clipping, lr, loss_name, validation_losses, hidden_state_size, task_name)
    if classify:
        plot_validation_acc(epochs, gradient_clipping, lr, loss_name, validation_accuracies, hidden_state_size,
                            task_name)


def validation(model, loss_layer, test_loader, validation_losses, device, classification, validation_accuracies,
               tensorboard_writer, epoch):
    total_loss = 0
    total_samples = 0 if classification else 1  # else to avoid div by 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_sequential = data.view(data.shape[0], 784, 1)
            if classification:
                resconstucted_batch, batch_pred_probs = model(data_sequential)
                loss = (loss_layer(data_sequential, resconstucted_batch) + model.cross_entropy(batch_pred_probs,
                                                                                               target)) / 2
                _, predicted = torch.max(batch_pred_probs.data, 1)
                total_samples += target.size(0)
                correct += (predicted == target).sum().item()
            else:
                resconstucted_batch = model(data_sequential)
                loss = loss_layer(data_sequential, resconstucted_batch)
                # the class with the highest energy is what we choose as prediction
            total_loss += loss
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


if __name__ == '__main__':
    results_path = os.path.join("/home", "mosesofe", "results", "pdl_Ass2")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.3081,))
    ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                               transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)

    hidden_state_sizes = [64]
    lrs = [0.001]
    gradient_clip = [1, 0]
    for lr in lrs:
        for clip in gradient_clip:
            for hidden_state_size in hidden_state_sizes:
                train(train_loader, test_loader, gradient_clipping=clip, hidden_state_size=hidden_state_size, lr=lr,
                      opt="adam")
