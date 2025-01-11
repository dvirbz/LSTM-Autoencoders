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
    _, n_rows, n_features = mnist_train.data.shape

    # mnist_train.data = mnist_train.data.view(-1, n_features)
    # mnist_test.data = mnist_test.data.view(-1, n_features)

    #criterion
    criterion = nn.MSELoss()

    optimizer_type = optim.Adam if args.optimizer == "adam" else optim.SGD # should expand to support more optimizers

    #grid search
    batch_size = args.batch_size
    hidden_size = n_features // 2 if args.hidden_size == 'half' else int(args.hidden_size)
    lr = args.lr
    gradient_clip = args.grad_clip
    epochs = args.epochs
    print(f"{mnist_train.data.shape=}, {mnist_test.data.shape=}")
        
    #create model
    model = LSTM_AE(n_features, hidden_size, bidirectional=args.bidirectional)
    print(model)

    #create data loaders
    train_loader = DataLoader(mnist_train.data, batch_size=batch_size)
    test_loader = DataLoader(mnist_test.data, batch_size=batch_size)

    #train
    train(train_loader, model, criterion, epochs, gradient_clip, lr, optimizer_type, device)
    
    #evaluate
    test_loss, outputs_pairs = evaluate(test_loader, model, criterion, device)
    print(f"Test loss: {test_loss}")
    test_loss = np.round(test_loss, 4)
    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}{'_FromGridSearch' if do_grid_search else ''}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")

    os.makedirs("./plots", exist_ok=True)

    n_digits_to_plot = 3
    #plot some outputs pairs
    plotted_digits = set()
    # mnist_test.data = mnist_test.data.view(-1, n_rows, n_features)

    for data, target in mnist_test:
        if target in plotted_digits:
            continue
        plotted_digits.add(target)
        input_img = data.reshape(n_rows, n_features)
        output = model(input_img.to(device)).detach().cpu().numpy()
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(input_img, cmap="gray")
        ax[0].set_title("True digit")
        ax[1].imshow(output, cmap="gray")
        ax[1].set_title("Predicted digit")
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.savefig(f"./plots/{model_name}_digit_{int(target)}.png")
        if len(plotted_digits) == n_digits_to_plot:
            break
        
if __name__ == "__main__":
    main()