import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

from argparser import get_train_args
from lstm_AE import LSTM_AE
from train_utils import train, evaluate
from utils import load_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = get_train_args()
    do_grid_search = args.grid_search
    
    print(args)
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    n_features = n_features

    mnist_train.data = mnist_train.data.view(-1, n_features)
    mnist_test.data = mnist_test.data.view(-1, n_features)

    #criterion
    criterion = nn.MSELoss()

    optimizer_type = optim.Adam if args.optimizer == "adam" else optim.SGD # should expand to support more optimizers

    best_params = None

    #grid search
    batch_size = best_params["batch_size"] if best_params else args.batch_size
    hidden_size = best_params["hidden_size"] if best_params else args.hidden_size
    lr = best_params["learning_rate"] if best_params else args.lr
    gradient_clip = best_params["grad_clip"] if best_params else args.grad_clip
    epochs = best_params["epochs"] if best_params else args.epochs
    print(f"{mnist_train.data.shape=}, {mnist_test.data.shape=}")
        
    #create model
    model = LSTM_AE(n_features, hidden_size)
    print(model)

    #create data loaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

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