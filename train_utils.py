import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class SqrtSched(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        return [base_lr / np.sqrt(self.last_epoch + 1) for base_lr in self.base_lrs]

def train_epoch_AE(train_loader, model, optimizer, criterion, grad_clip, device):
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.float().to(device)
        output = model(data).to(device)
        loss = criterion(output, data)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # print(f"grads: {optimizer.param_groups[0]['params'][0].grad}")
        optimizer.step()
        running_loss += loss.item()

    total_loss = running_loss / len(train_loader)
    return total_loss

def train_epoch_CLS(train_loader, model, optimizer, criterion, grad_clip, device):
    running_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    for data, targets in train_loader:
        optimizer.zero_grad()
        targets = one_hot(targets, 10).to(torch.float32).to(device) # fix to generalize            
        data = data.float().to(device)
        output, probs = model(data)
        output = output.to(device)
        probs = probs.to(device)
        ce_loss = ce_criterion(probs, targets)
        ae_loss = criterion(output, data)
        loss = ae_loss + ce_loss
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()

    total_loss = running_loss / len(train_loader)
    return total_loss

def evaluate_AE(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.float().to(device)
            output = model(data).to(device)
            all_outputs.append((data, output))
            loss = criterion(output, data)
            accumulative_loss += loss.item()

    total_loss = accumulative_loss / len(val_loader)
    return total_loss, all_outputs

def evaluate_CLS(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for data, targets in val_loader:
            labels.extend(targets.to(device))
            targets = one_hot(targets, 10).to(torch.float32).to(device) # fix to generalize
            data = data.float().to(device)
            
            output, probs = model(data)          
            output = output.to(device)
            probs = probs.to(device)
            
            curr_preds = torch.argmax(probs, dim=1)
            preds.extend(curr_preds)
            
            ce_loss = ce_criterion(probs, targets)
            ae_loss = criterion(output, data)
            loss = ce_loss + ae_loss
            accumulative_loss += loss.item()

    accuracy = (torch.tensor(labels) == torch.tensor(preds)).sum().item() / len(labels)
    print(f"{accuracy=}")
    total_loss = accumulative_loss / len(val_loader)
    return total_loss, accuracy

def train(train_loader,
          model,
          criterion,
          epochs,
          grad_clip,
          learning_rate,
          optimizer_type,
          device,
          is_cls_ae = False):
    model.to(device)
    model.train()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        if is_cls_ae:
            train_loss = train_epoch_CLS(train_loader, model, optimizer, criterion, grad_clip, device)
        else:   
            train_loss = train_epoch_AE(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        tqdm.write(f"Epoch: {epoch}, Loss: {train_loss}")
    
    return train_loss

def plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      grad_clip,
                      learning_rate,
                      optimizer_type,
                      save_path,
                      device,
                      is_cls_ae = False):
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
    def update_graph(train_loss, learning_rate):
        train_losses.append(train_loss)
        train_line.set_data(range(1, len(train_losses) + 1), train_losses)
        ax.relim()
        ax.autoscale_view()
        loss_text.set_text(f'Train Loss: {train_loss:.4f}')
        lr_text.set_text(f'Learning Rate: {learning_rate:.5e}')
        plt.draw()
        plt.pause(0.01)

    model.to(device)
    model.train()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    print(f"Optimizer: {optimizer}")
    scheduler = SqrtSched(optimizer)
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        if is_cls_ae:
            train_loss = train_epoch_CLS(train_loader, model, optimizer, criterion, grad_clip, device)
        else:   
            train_loss = train_epoch_AE(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        update_graph(train_loss, optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.savefig(f'{save_path}_loss_plots.png')
    plt.show()

    print(f"Training complete.\nModel final loss: {train_losses[-1]:.2f}")
