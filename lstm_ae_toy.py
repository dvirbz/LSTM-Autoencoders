import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import itertools

from argparser import get_train_args
from lstm_AE import LSTM_AE

def load_data(data_folder):
    X_train = np.load(f"{data_folder}/X_train.npy")
    X_val = np.load(f"{data_folder}/X_val.npy")
    X_test = np.load(f"{data_folder}/X_test.npy")
    return X_train, X_val, X_test

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


def grid_search(X_train, X_val, criterion, optimizer_type, device) -> dict:
    #grid search
    hidden_sizes = [25, 30, 35]
    grad_clips =  [1] #[1, 5]
    learning_rates = [0.001, 0.01]
    batch_sizes = [32, 64]
    Epochs = [200, 1000]

    best_params = {
        "hidden_size": None,
        "grad_clip": None,
        "learning_rate": None,
        "batch_size": None,
        "epochs": None,     
    }
    best_val_loss = float("inf")
    best_trial = None
    for i, (ep, hs, gc, lr, bs) in \
    tqdm(enumerate(itertools.product(Epochs, hidden_sizes, grad_clips, learning_rates, batch_sizes)), desc="Grid search"):
        tqdm.write(f"Starting with parameters hs: {hs}, gc: {gc}, lr: {lr}, bs: {bs}, ep: {ep}")
        model = LSTM_AE(X_train.shape[1], hs).to(device)
        train_loader = DataLoader(X_train, batch_size=bs, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=bs, shuffle=False)
        train_loss = train(train_loader, model, criterion, ep, gc, lr, optimizer_type, device)
        val_loss, _ = evaluate(val_loader, model, criterion, device)
        tqdm.write(f"train_loss: {train_loss}, val_loss: {val_loss}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params["hidden_size"] = hs
            best_params["grad_clip"] = gc
            best_params["learning_rate"] = lr
            best_params["batch_size"] = bs
            best_params["epochs"] = ep
            best_trial = i

        tqdm.write(f"best parameters: {best_params} with val_loss: {best_val_loss} was found in trial {best_trial}")
    return best_params

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = get_train_args()
    data_folder = args.data_folder
    do_grid_search = args.grid_search
    
    print(args)
    if data_folder:
        print(data_folder)        
        #load 
        X_train, X_val, X_test = load_data(data_folder)
        print(f"shape x_train: {X_train.shape}, shape x_val: {X_val.shape}, shape x_test: {X_test.shape}")

    #criterion
    criterion = nn.MSELoss()

    optimizer_type = optim.Adam if args.optimizer == "adam" else optim.SGD # should expand to support more optimizers

    best_params = None

    #grid search
    if do_grid_search:
        best_params = grid_search(X_train, X_val, criterion, optimizer_type, device)
        print(f"Best parameters: {best_params}")
    batch_size = best_params["batch_size"] if best_params else args.batch_size
    hidden_size = best_params["hidden_size"] if best_params else args.hidden_size
    lr = best_params["learning_rate"] if best_params else args.lr
    gradient_clip = best_params["grad_clip"] if best_params else args.grad_clip
    epochs = best_params["epochs"] if best_params else args.epochs
        
    #create model
    model = LSTM_AE(X_train.shape[1], hidden_size)
    print(model)

    #create data loaders
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    #train
    train(train_loader, model, criterion, epochs, gradient_clip, lr, optimizer_type, device)
    
    #evaluate
    test_loss, outputs_pairs = evaluate(test_loader, model, criterion, device)
    print(f"Test loss: {test_loss}")
    test_loss = np.round(test_loss, 4)
    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}{'_FromGreadSearch' if do_grid_search else ''}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    
    #plot some outputs pairs
    for i in range(2):
        data, output = outputs_pairs[i]
        data = data[0] 
        output = output[0]
        print(data.shape, output.shape)
        plt.subplot(2, 1, i+1)
        plt.plot(data.detach().numpy(), label="data")
        plt.plot(output.detach().numpy(), label="output")
        plt.legend()
        plt.title(f"Output pair {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{model_name}.png")
        
if __name__ == "__main__":
    main()