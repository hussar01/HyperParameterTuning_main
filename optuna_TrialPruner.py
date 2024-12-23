import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from optuna_libNCV import MIN_RESOURCE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import optuna
import numpy as np
import matplotlib.pyplot as plt
#from optuna.visualization import plot_contour
#from optuna.visualization import plot_edf
#from optuna.visualization import plot_intermediate_values
#from optuna.visualization import plot_optimization_history
#from optuna.visualization import plot_parallel_coordinate
#from optuna.visualization import plot_param_importances
#from optuna.visualization import plot_rank
#from optuna.visualization import plot_slice
#from optuna.visualization import plot_timeline
#import plotly
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
'''--------------------------------------------'''
from functools import partial
import os
import sys
import tempfile
from pathlib import Path
from sched import scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import  time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from contextlib import contextmanager
import ssl
from datetime import datetime


# Constants
# Constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# If using GPU, you can also print the GPU name
if device.type == 'cuda':
    print("GPU is available\n")
    print("GPU:", torch.cuda.get_device_name(0))

# check if a GPU is available
'''with_gpu = torch.cuda.is_available()
if with_gpu:
    DEVICE = torch.device("cuda")
    print("GPU is available\n")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available\n")'''


EPOCHS = 24
CLASSES = 10
INNER_FOLD = 5
NUM_SAMPLES = 24
DIR = os.getcwd()

#-------PRUNER SETTING------------
MIN_RESOURCE = 8      #till this no of epoch the pruning will not happen for the current fold
REDUCTION_FACTOR = 2  #responisble for double resource and halving the trials at each rung
#-------TPE SETTING-------------
N_STARTUP_TRIALS = 8 #The random sampling is used instead of the TPE algorithm until the given number of trials finish in the same study.


# Model Definition
class Net(nn.Module):
    def __init__(self, l1, l2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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

def calculate_overfitting(train_loss, val_loss):
    """Compute overfitting metric as difference between train and validation loss."""
    return val_loss - train_loss

#INSTANCE_ID = f"{os.getpid()}_{int(time.time())}"
current_time = datetime.now().strftime("%Y%m%d")
INSTANCE_ID = f"{os.getpid()}_{current_time}"
INSTANCE_DIR = os.path.join("instances", INSTANCE_ID)
os.makedirs(INSTANCE_DIR, exist_ok=True)
print(f"Script instance is using directory: {INSTANCE_DIR}")

def train_mnist(trial, config, train_loader, val_loader, max_num_epochs):
    model = Net(config["l1"], config["l2"]).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_cmp = float('inf')  # Track best validation loss
    best_checkpoint_path = None

    for epoch in range(max_num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

        train_loss = running_loss / len(train_loader)
        print(f"**STATS for Epoch {epoch + 1}** : ")
        print(f"Average training loss: {train_loss:.4f}")
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(f"Average validation loss: {val_loss:.4f}")
        val_acc = validation_accuracy(model, val_loader, DEVICE)
        print(f"Validation Accuracy: {val_acc:.4f}")
        overfitting = calculate_overfitting(train_loss, val_loss)
        print(f"Overfitting: {overfitting:.4f}")
        #composite_metric = train_loss + overfitting
        #print(f"Composite Metric: {composite_metric:.4f}")

        # Report metrics to Optuna at each epoch for intermediate evaluation
        trial.report(val_loss, step=epoch) #changing composite metric to train loss as metric for optuna and pruning

        # Implement epoch-level early stopping based on patience (optional)
        if trial.should_prune():
            raise optuna.TrialPruned()
            #print(f"Early stopping epoch {epoch + 1} for trial {trial.number + 1}. Moving to next fold.")
            #break
            #raise optuna.exceptions.TrialPruned()

        # Check if this is the best model based on validation loss
        if val_loss < best_cmp:
            best_cmp = val_loss
            checkpoint_dir = os.path.join(INSTANCE_DIR,f"best_checkpoint_trial_{trial.number}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {best_cmp:.4f}")

    return val_loss #train_mnist function will report metric training loss

def train_mnist_final(trial, config, train_loader, te_loader, max_num_epochs):
    model = Net(config["l1"], config["l2"]).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_cmp = float('inf')  #Track best validation loss
    best_checkpoint_path = None
    train_losses = []
    val_losses = []

    for epoch in range(max_num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

        train_loss = running_loss / len(train_loader)
        print(f"**STATS for Epoch {epoch + 1}** : ")
        print(f"Average training loss: {train_loss:.4f}")
        val_loss = test_loss(model, te_loader, criterion, DEVICE)
        print(f"Average validation loss: {val_loss:.4f}")
        #val_acc = test_accuracy(model, te_loader, DEVICE)
        #print(f"Validation Accuracy: {val_acc:.4f}")
        overfitting = calculate_overfitting(train_loss, val_loss)
        print(f"Overfitting: {overfitting:.4f}")
        #composite_metric = train_loss + overfitting
        #print(f"Composite Metric: {composite_metric:.4f}")

        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Report metrics to Optuna at each epoch for intermediate evaluation
        trial.report(train_loss, step=epoch) #changing composite metric to train loss as metric for optuna and pruning

        # Implement epoch-level early stopping based on patience (optional)

        # Check if this is the best model based on validation loss
        if train_loss < best_cmp:
            best_cmp = train_loss
            checkpoint_dir = os.path.join(INSTANCE_DIR,f"best_checkpoint_trial_{trial.number}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best model saved at epoch {epoch + 1} with training loss: {best_cmp:.4f}")

    plot_training_validation_loss(range(1, max_num_epochs + 1), train_losses, val_losses)

    return train_loss, val_loss, overfitting #train_mnist function will report metric training loss

def plot_training_validation_loss(epochs, train_losses, val_losses):
    """
    Plots training and validation loss per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig("Results/TPE/Ranges_dec_11/SHP/SHP_Ranges_split5_2.png")

def validate(model, valloader, criterion, DEVICE):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(valloader)
    model.train()
    return val_loss

def test_loss(model,testloader ,criterion, DEVICE):
    model.eval()
    '''_, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )'''
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(testloader)
    #model.train()
    return test_loss

def test_accuracy(model, testloader,  DEVICE):
    model.eval()
    '''_, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )'''

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    #print(f"Test set accuracy: {test_acc:.4f}")
    return test_acc


def validation_accuracy(model, valloader, DEVICE):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# Objective function for hyperparameter optimization
def objective(trial):
    # Create hyperparameter config using trial suggestions

    # Suggest hyperparameters using the trial instance
    #l1 = trial.suggest_int("l1", 64, 512, step=32)
    #l1 = trial.suggest_categorical('l1', [128, 256])
    #l2 = trial.suggest_int("l2", 32, l1, step=32)  # Use l1's value as the upper bound for l2
    #l2 = trial.suggest_categorical('l2', [64, 128])
    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])


    #config = {
        #"l1": l1,
        #"l2": l2,
        #"lr": lr,
        #"batch_size": batch_size
    #}


    '''config = {
        "l1": optuna.trial.Trial.suggest_int("l1", 64, 512, step=32),
        "l2": optuna.trial.Trial.suggest_int("l2", 32, 'l1', step=32),
         "lr" : optuna.trial.Trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        "batch_size": optuna.trial.Trial.suggest_categorical("batch_size", [16,32, 64,128,256])
    }'''

    config = {
        "l1": trial.suggest_categorical("l1", [128, 256]),
        "l2": trial.suggest_categorical("l2", [64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16,32, 64,128,256])
    }

    '''config = {
        "l1": trial.suggest_categorical("l1", [128, 256]),
        "l2": trial.suggest_categorical("l2", [64, 128]),
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64])
    }'''

    print(f"\nSelected Hyperparameters for Trial {trial.number + 1}:")
    print(f"  l1: {config['l1']}, l2: {config['l2']}, lr: {config['lr']}, batch_size: {config['batch_size']}")

    trainset, _ = load_data()  # Load the dataset
    print(f"Training set size: {len(trainset)}")

    # K-Fold cross-validation setup
    kf = KFold(n_splits=INNER_FOLD, shuffle=True, random_state=0)
    #composite_metrics = []  all training losses of each fold will be stored in this array previously used for storing composite metrics
    val_losses = []  #all validation losses of each fold will be stored in this array previously used for storing composite metrics
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(trainset)))):
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")
        inner_train = torch.utils.data.Subset(trainset, train_idx)
        inner_val = torch.utils.data.Subset(trainset, val_idx)
        print(f"Inner Training set size for fold {fold + 1} is {len(inner_train)}")
        print(f" Inner Validation set size for fold {fold + 1} is {len(inner_val)}")

        train_loader = DataLoader(inner_train, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(inner_val, batch_size=config["batch_size"], shuffle=True)

        #print(f"Validation set size: {len(val_loader)}")
        #print(f"Training set size: {len(train_loader)}")

        val_loss = train_mnist(trial,config, train_loader, val_loader, max_num_epochs=EPOCHS)
        val_losses.append(val_loss)
        print(f"Fold {fold + 1} validation loss: {val_loss:.4f}")

    mean_validation_loss = np.mean(val_losses)
    print(f"Mean validation loss across all folds for Trial {trial.number + 1} is {mean_validation_loss:.4f} with trial config:  l1: {config['l1']}, l2: {config['l2']}, lr: {config['lr']}, batch_size: {config['batch_size']}")
    return mean_validation_loss


# Run the Optuna optimization
def run_optuna(num_samples=NUM_SAMPLES):
    start_time = time.time()
    print("INSTANT NO 10")
    print(f"Total number of trial for this search space: {num_samples}")
    study = optuna.create_study(
        storage="sqlite:///ashapruner_SHP_Ranges_split5_2.db",
        study_name="asha_pruner_SHP_Ranges_split5_2",
        load_if_exists=False,
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=MIN_RESOURCE, reduction_factor= REDUCTION_FACTOR),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=N_STARTUP_TRIALS, multivariate=True, group=True),
    )
    study.optimize(objective, n_trials=num_samples)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best hyperparameters found:")
    print(study.best_trial.params)

    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)

    #loading full train set and test set , loaddata() return these datasets
    trainset, testset = load_data()
    #Train on full train set
    best_model = Net(best_trial.params["l1"],best_trial.params["l2"]).to(DEVICE)
    optimizer = optim.SGD(best_model.parameters(), lr=best_trial.params["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    best_model.to(DEVICE)



    # Load the best checkpoint
    best_checkpoint_path = os.path.join(
        INSTANCE_DIR, f"best_checkpoint_trial_{best_trial.number}/model.pth")
    if os.path.exists(best_checkpoint_path):
        best_model.load_state_dict(torch.load(best_checkpoint_path))
        print(f"Loaded best model checkpoint from: {best_checkpoint_path}")
    else:
        print("No checkpoint found for the best trial")

    # Train and test on  full train and test
    full_train_loader = DataLoader(trainset, batch_size=best_trial.params["batch_size"], shuffle=True)
    full_test_loader =  DataLoader(testset, batch_size=best_trial.params["batch_size"], shuffle=True)
    #best_model = Net(best_trial.params["l1"], best_trial.params["l2"]).to(DEVICE)
    #optimizer = optim.SGD(best_model.parameters(), lr=best_trial.params["lr"], momentum=0.9)
    #criterion = nn.CrossEntropyLoss()
    print(f"Using best hyperparameters {study.best_trial.params} on final Train set with train set size : {len(trainset)}")
    Train_loss, Test_loss, Overfitting = train_mnist_final(best_trial, best_trial.params, full_train_loader, full_test_loader, max_num_epochs=EPOCHS)
    print("+++FINAL STATS++++")
    print(f"Training Loss {Train_loss}")
    #Calculating test loss on final test set)
    print(f"Using best hyperparameters {study.best_trial.params} on final Test set to find Test loss for overfitting")
    #testing_loss = test_loss(best_model, full_test_loader, criterion, DEVICE)
    print(f" Testing loss : {Test_loss:.4f}")
    #Calculating overfitting
    #Overfitting = calculate_overfitting(final_train_set, testing_loss)
    print(f"Calculated Overfitting : {Overfitting:.4f}")
    #Test on unseen test set
    print(f"Using best hyperparameters {study.best_trial.params} on final Test set with testing set size : {len(testset)}")
    test_accuracy_value = test_accuracy(best_model,full_test_loader, DEVICE)
    print(f"Test set accuracy with best hyperparameters: {test_accuracy_value:.4f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Calculate hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total time taken for hyperparameter tuning and evaluation: {hours}:{minutes}:{seconds}")


    #plt.show(block=True)
    #plt.interactive(False)

    fig1 = optuna.visualization.plot_optimization_history(study)
    #fig1.show()

    fig1.write_image("Results/TPE/Ranges_dec_11/SHP/optimization_history_SHP_Ranges_split5.png")


    fig2 = optuna.visualization.plot_timeline(study)
    #fig2.show()

    fig2.write_image("Results/TPE/Ranges_dec_11/SHP/timeline_SHP_Ranges_split5.png")




if __name__ == "__main__":
    run_optuna(num_samples=NUM_SAMPLES)
