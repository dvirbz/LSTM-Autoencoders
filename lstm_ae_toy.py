import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from argparser import get_train_args
from lstm_AE import LSTM_AE
from train_utils import plot_train_losses, evaluate, optuna_train
from utils import load_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = get_train_args()
    data_folder = args.data_folder
    hyper_search = args.hyper_search
    
    if data_folder:
        print(data_folder)        
        X_train, X_val, X_test = load_data(data_folder)
        print(f"shape x_train: {X_train.shape}, shape x_val: {X_val.shape}, shape x_test: {X_test.shape}")

    #criterion
    criterion = nn.MSELoss()

    optimizer_type = optim.Adam if args.optimizer == "adam" else optim.SGD

    #create data loaders
    batch_size = args.batch_size
    n_epochs = args.epochs
    input_shape = X_train.shape[1]
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    #optuna search
    best_params = None
    if hyper_search:
        n_trials = 100
        best_params = optuna_train(train_loader, val_loader, criterion, optimizer_type, args.trial_epochs, 'ae', n_trials, device)

    hidden_size = best_params["hidden_size"] if best_params else args.hidden_size
    hidden_size = input_shape // 2 if hidden_size == 'half' else int(hidden_size)
    lr = best_params["learning_rate"] if best_params else args.lr
    gradient_clip = best_params["grad_clip"] if best_params else args.grad_clip
    bidirectional = args.bidirectional
        
    #create model
    model = LSTM_AE(input_shape, hidden_size, bidirectional=bidirectional)
    print(model)

    #train
    save_path = "./plots/toy/"
    plot_train_losses(train_loader,
                      model,
                      criterion,
                      n_epochs,
                      gradient_clip,
                      lr,
                      optimizer_type,
                      save_path,
                      'ae',
                      device,
                      )
    
    
    #evaluate
    test_loss, outputs_pairs = evaluate(test_loader, model, criterion, device)
    print(f"Test loss: {test_loss}")
    test_loss = np.round(test_loss, 4)

    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}{'_FromGreadSearch' if do_grid_search else ''}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    for i in range(2):
        data, output = outputs_pairs[i]
        data = data[0] 
        output = output[0]
        print(data.shape, output.shape)
        plt.subplot(2, 1, i+1)
        plt.plot(data.detach().cpu().numpy(), label="data")
        plt.plot(output.detach().cpu().numpy(), label="output")
        plt.legend()
        plt.title(f"Output pair {i}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{model_name}.png")
        
if __name__ == "__main__":
    main()