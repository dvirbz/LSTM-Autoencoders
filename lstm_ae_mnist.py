import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from argparser import get_train_args
from lstm_AE import LSTM_AE_Classifier
from train_utils import train, evaluate, plot_train_losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = get_train_args()

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.squeeze()),
    ])
    if args.pixel_wise:
        # add transformation to flatten the image
        transformations = transforms.Compose(transformations.transforms +
                                             [transforms.Lambda(lambda x: x.view(-1, 1))])

    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transformations)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transformations)
    _, n_rows_og, n_features_og = mnist_train.data.shape
    n_rows = n_rows_og if not args.pixel_wise else n_rows_og * n_features_og
    n_features = n_features_og if not args.pixel_wise else 1
    print(f"{n_rows=}, {n_features=}")

    #criterion
    criterion = nn.MSELoss()

    optimizer_type = optim.Adam if args.optimizer == "adam" else optim.SGD

    #grid search
    batch_size = args.batch_size
    hidden_size = n_features // 2 if args.hidden_size == 'half' else int(args.hidden_size)
    lr = args.lr
    gradient_clip = args.grad_clip
    epochs = args.epochs
        
    #create data loaders
    test_full_loader = DataLoader(mnist_test, batch_size=1)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)
    
    #create model
    # model = LSTM_AE(n_features, hidden_size, bidirectional=args.bidirectional)
    model = LSTM_AE_Classifier(n_features, hidden_size, len(np.unique(mnist_train.targets)) ,bidirectional=args.bidirectional)
    print(model)

    # Make sure correct plot folder exits
    pixel_wise = "_Pixel_wise" if args.pixel_wise else ""
    plot_folder = "./plots/MNIST"
    os.makedirs(plot_folder, exist_ok=True)
    plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      gradient_clip,
                      lr,
                      optimizer_type,
                      f'{plot_folder}/'+ pixel_wise,
                      'cls',
                      device,
                      )


    
    #evaluate
    test_loss, accuracy = evaluate(test_loader, model, criterion, device, is_cls_ae=True)
    print(f"Test loss: {test_loss}")
    test_loss = np.round(test_loss, 4)

    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}{'_FromGridSearch' if do_grid_search else ''}{pixel_wise}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    n_digits_to_plot = 3
    #plot some outputs pairs
    plotted_digits = set()

    for data, target in test_full_loader:
        if target in plotted_digits:
            continue
        plotted_digits.add(target)
        output, probs = model(data.to(device))
        input_img = data.squeeze().detach().cpu().numpy().reshape(n_rows_og, n_features_og)
        output = output.squeeze().detach().cpu().numpy().reshape(n_rows_og, n_features_og)
        predicted_digit = torch.argmax(probs)
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(input_img, cmap="gray")
        ax[0].set_title("True digit")
        ax[1].imshow(output, cmap="gray")
        ax[1].set_title(f"Predicted digit {predicted_digit}")
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.savefig(f"./plots/{model_name}_digit_{int(target)}.png")
        if len(plotted_digits) == n_digits_to_plot:
            break
        
if __name__ == "__main__":
    main()