import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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
    for _ in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, grad_clip, device)
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