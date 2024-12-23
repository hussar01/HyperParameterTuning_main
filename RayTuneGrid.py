from functools import partial
import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler
#from ray.tune.search.random import RandomSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from contextlib import contextmanager
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
#writer = SummaryWriter(log_dir=None)

# Configure logging (if necessary)
#log_dir = r"C:\Users\hussain\PycharmProjects\logs"
#os.makedirs(log_dir, exist_ok=True)
# Uncomment this line if you need the SummaryWriter for other logging purposes
#writer = SummaryWriter(log_dir=log_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    with suppress_stdout():
        trainset = datasets.MNIST(root="./data",train=True,download=True,transform=transform) # You can keep this default to ensure data is downloaded to a known location
        testset = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

    return trainset, testset


# Load the MNIST data directly
trainset, testset = load_data()

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Change input channels from 3 to 1 for grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, l1)  # Adjusted based on input image size (28x28)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_overfitting(train_loss, val_loss):
    """Compute overfitting metric as difference between train and validation loss."""
    return val_loss - train_loss

def train_mnist(config, data_dir=None):
    net = Net(config["l1"], config["l2"])

    #device = DEVICE  # Force device to be CPU
    net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Print the selected learning rate before starting the training
    print(f"Selected learning rate: {config['lr']}")
    # Print configuration to monitor selected hyperparameters
    print(f"Using config: l1={config['l1']}, l2={config['l2']}, lr={config['lr']}, batch_size={config['batch_size']}")

    # Load from checkpoint if available
    checkpoint: Checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                net.load_state_dict(checkpoint['net_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Checkpoint loaded from {checkpoint_path}")

    trainset, testset = load_data()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    num_epochs = 10

    for epoch in range(num_epochs):  # Ensure this loop only runs for one epoch
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if i % 2000 == 1999:  # print every 2000 mini-batches
            # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            # running_loss = 0.0
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

        # Calculating test accuracy after each epoch using test data
        accuracy = test_accuracy(net, DEVICE) #I am using test set as validation set during training
        # Calculate accuracy and validation loss after each epoch
        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = validate(net, testloader, criterion, DEVICE)

        # Calculate overfitting metric (difference between validation and training loss)
        overfitting = calculate_overfitting(avg_train_loss, avg_val_loss)

        # Composite metric: combine loss and overfitting
        composite_metric = avg_train_loss + overfitting  # Example, can be customized

        print("Saving checkpoint...")
        checkpoint_dir = f"checkpoint_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the checkpoint directory exists
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pth"
        torch.save({
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

        '''
        # Save the model state as a checkpoint at the end
        checkpoint = {
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),}
        # Save the model state as a checkpoint at the end of the epoch using Ray Tune's checkpoint_dir
        #with train.Checkpoint.from_directory(str(epoch)) as checkpoint_dir:'''

        # Report the loss and accuracy metrics using ray.train.report
        #train.report({"loss": running_loss / len(trainloader), "accuracy": accuracy},
                     #checkpoint=Checkpoint.from_directory(checkpoint_dir))

        # Report the loss, accuracy, and overfitting metrics using ray.train.report
        train.report({
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "overfitting": overfitting,
            "composite_metric": composite_metric  # Report composite metric
        }, checkpoint=Checkpoint.from_directory(checkpoint_dir))

    '''for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Report the loss metric to Ray Tune
            running_loss += loss.item()

            # Report the loss metric using ray.train.report
            train.report({"loss": running_loss / (i + 1)}) 
            #train.report(loss=running_loss / (i + 1))  # Average loss per mini-batch

            #running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0'''


def validate(net, testloader, criterion, device=DEVICE):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(testloader)
    net.train()
    return val_loss

def test_accuracy(net, device=DEVICE):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


import ray
from ray import tune


def main( max_num_epochs=24):
    #writer = SummaryWriter(log_dir=None)
    ray.shutdown()
    ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False)

    config = {
        # "l1": tune.choice([2**i for i in range(9)]),
        # "l2": tune.choice([2**i for i in range(9)]),
        "l1": tune.grid_search([128, 256]),
        "l2": tune.grid_search([64, 128]),
        #"lr": tune.loguniform(1e-4, 1e-2),
        "lr": tune.grid_search([1e-3, 5e-4, 1e-4]),
        "batch_size": tune.grid_search([32, 64]),
    }


    scheduler = AsyncHyperBandScheduler(   #introducing early stopping and successive halving
        metric="composite_metric",
        mode="min",
        max_t=max_num_epochs,  #max amount of epochs to run if a trial performs good and is not early stopped
        grace_period=8, #least amount of epochs to run for each trial before early stoppping
        reduction_factor=2, #half of the trials are stopped at each halving stage, while the best-performing half continues.
                            # Each halving stage occurs after a predefined number of resource units has been consumed.
                            # Stop more trials early (about 2/3 of trials will be stopped after each halving)
    )
    # ray.init(ignore_reinit_error=True)
    result = tune.run(
        train_mnist,  # No need to pass data_dir anymore
        #resources_per_trial={"cpu": 5, "gpu": gpus_per_trial},
        config=config,
        scheduler=scheduler,
        # change 1 adding verbose for report didnt work
        verbose=1,
        #loggers = []
        #progress_reporter=None,
        #log_to_file = False
        # checkpoint_at_end=True
    )

    best_trial = result.get_best_trial("composite_metric", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final training loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial final overfitting: {best_trial.last_result['overfitting']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    #device = "cpu"
    best_trained_model.to(DEVICE)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    if best_checkpoint:
        with best_checkpoint.as_directory() as checkpoint_dir:
            # data_path = Path(checkpoint_dir) / "data.pkl"
            data_path = Path(checkpoint_dir) / "checkpoint.pth"
            checkpoint = torch.load(data_path)
            best_trained_model.load_state_dict(checkpoint["net_state_dict"])
            test_acc = test_accuracy(best_trained_model, DEVICE)
            print("Best trial test set accuracy: {}".format(test_acc))
            # with open(data_path, "rb") as fp:
            # best_checkpoint_data = pickle.load(fp)

        # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        # test_acc = test_accuracy(best_trained_model, device)
        # print("Best trial test set accuracy: {}".format(test_acc))
    else:
        print("No checkpoint found for the best trial")

    '''best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    if best_checkpoint:
        with best_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pth"
            checkpoint = torch.load(checkpoint_path)

            best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
            optimizer = torch.optim.SGD(best_trained_model.parameters(), lr=best_trial.config["lr"], momentum=0.9)
            best_trained_model.load_state_dict(checkpoint['net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            test_acc = test_accuracy(best_trained_model, device)
            print("Best trial test set accuracy: {}".format(test_acc))
    else:
        print("No checkpoint found for the best trial")'''

    # Shutdown Ray after running
    ray.shutdown()


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main( max_num_epochs=24)