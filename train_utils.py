import torch.nn as nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


def train_epoch(train_loader, model, optimizer, criterion, grad_clip, device):
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.float().to(device)
        output = model(data).to(device)
        loss = criterion(output, data)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def train(train_loader, model, criterion, epochs, grad_clip, learning_rate, optimizer_type, device):
    model.to(device)
    model.train()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        # tqdm.write(f"Epoch: {epoch}, Loss: {train_loss}")
    
    return train_loss

def evaluate(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    all_outputs = []
    for data in val_loader:
        data = data.float().to(device)

        output = model(data).to(device)
        all_outputs.append((data, output))

        loss = criterion(output, data)
        accumulative_loss += loss.item()

    return accumulative_loss / len(val_loader), all_outputs


def plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      grad_clip,
                      learning_rate,
                      optimizer_type,
                      save_path,
                      device):
    """
    Train a neural network model and display an interactive graph of training
    and validation losses.

    Args:
        model (NeuralNetwork): The neural network model to train.
        loss_fn (Loss): The loss function used for training.
        optim (Optimizer): The optimizer used for parameter updates.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        X_train (np.ndarray): Training data features.
        C_train (np.ndarray): Training data labels.
        X_val (np.ndarray): Validation data features.
        C_val (np.ndarray): Validation data labels.
    """
    # Initialize variables to store losses
    train_losses = []

    # Prepare the interactive plot
    _, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    train_line, = ax.plot([], [], label='Train Loss', color='blue')
    val_line, = ax.plot([], [], label='Validation Loss', color='orange')
    plt.legend()

    # Add a text box for learning rate
    lr_text = ax.text(0.1, 0.95, '', transform=ax.transAxes)
    loss_text = ax.text(0.1, 0.85, '', transform=ax.transAxes)

    # Helper function to update the graph
    def update_graph(train_loss):
        train_losses.append(train_loss)
        train_line.set_data(range(len(train_losses)), train_losses)
        ax.relim()
        ax.autoscale_view()
        loss_text.set_text(f'Train Loss: {train_loss:.2f}')
        plt.draw()
        plt.pause(0.01)

    model.to(device)
    model.train()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        update_graph(train_loss)

    plt.savefig(f'{save_path}_loss_plots.png')
    plt.show()

    print(f"Training complete.\nModel final loss: {train_losses[-1]:.2f}")
