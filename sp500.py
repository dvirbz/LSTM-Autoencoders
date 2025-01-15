import os
import torch
from torch import nn
from torchvision import transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from argparser import get_train_args
from lstm_AE import LSTM_AE, LSTM_AR
from train_utils import evaluate, plot_train_losses

def plot_daily_max(df, stocks):
    plt.figure(figsize=(12, 8))
    for stock in stocks:
        plt.subplot(len(stocks), 1, stocks.index(stock) + 1)
        sns.lineplot(data=df[df["symbol"] == stock], x="date", y="high")
    plt.show()    
    
def preprocess_data(df):
    # Clean Data
    df.dropna(inplace=True)
    df.drop(columns=["volume"], inplace=True)
    # Drop stocks with less than the maximum amount of days
    stock_days = df.groupby("symbol")["date"].count()
    max_days = stock_days.max()
    stocks_to_remove = stock_days[stock_days < max_days].index
    df = df[~df["symbol"].isin(stocks_to_remove)]
    
    df_pivot = df.pivot(index="symbol", columns="date", values=["open", "high", "low", "close"])
    # Reorder the columns to match the desired shape (number of symbols, number of dates, 4)
    df_pivot = df_pivot.swaplevel(axis=1).sort_index(axis=1)
    # Convert the dataframe to a numpy array
    dataset = df_pivot.values.reshape(len(df_pivot), max_days, 4)
    return dataset

def create_autoregressive_data(data):
    # Create an autoregressive dataset
    # assuming shpae is (Batches, Sequence Length, Features) - (B, Amount of Days, 4)
    y = data[:, 1:, :]
    X = data[:, :-1, :]
    return torch.tensor(X), torch.tensor(y)

def split_data(dataset):
    N_samples, N_days, N_features = dataset.shape
    print(f"{dataset.shape=}")
    dataset = torch.tensor(dataset, dtype=torch.float32)

    # dataset = dataset.permute(1, 0, 2)
    train_data, test_data = train_test_split(dataset.detach().cpu(), test_size=0.2, shuffle=True)
    train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=True)

    train_min, train_max = train_data.min(), train_data.max()

    first_transform = transforms.Lambda(lambda x: torch.tensor(x,dtype=torch.float32)) if args.auto_regressor

    transform = transforms.Compose([
        first_transform,
        transforms.Lambda(lambda x: (x - train_min) / (train_max - train_min))
        ])

    train_data = transform(train_data)
    val_data = transform(val_data)
    test_data = transform(test_data)

    print(f"{train_data.shape=}\n{val_data.shape=}\n{test_data.shape=}")

    return train_data, val_data, test_data

def plot_ae(test_loader, model, number_of_plots=3):
    i = 0
    for data in test_loader:
        if i == number_of_plots:
            break
        output = model(data)
                
        open, high, low, close = data.squeeze().permute(1, 0).detach().cpu().numpy()
        pred_open, pred_high, pred_low, pred_close = output.squeeze().permute(1, 0).detach().cpu().numpy()
        plt.subplot(2, 2, 1)
        plt.plot(open, label="Open")
        plt.plot(pred_open, label="Pred Open")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(high, label="High")
        plt.plot(pred_high, label="Pred High")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(low, label="Low")
        plt.plot(pred_low, label="Pred Low")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(close, label="Close")
        plt.plot(pred_close, label="Pred Close")
        plt.legend()
        plt.show()
        i += 1

def plot_ar(test_loader, model, number_of_plots=3):
    i = 0
    for data, target in test_loader:
        x_hat, y_hat = model(data)
        
def main():
    data_path = "./data/"
    file_name = "sp500.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # plot_daily_max(df, ["AMZN", "GOOGL"])
    
    # Dataset Shape - (Batches, Sequence Length, Features) - (B, Amount of Days, 4)
    # Features: Open, High, Low, Close
    # Targets: Open, High, Low, Close
    
    # print(df[df.isna().any(axis=1)])
    dataset = preprocess_data(df)
    # if args.auto_regressor:
    #     dataset, dataset_y = create_autoregressive_data(dataset)
    
    # # 1 1 1 1 1 1 0 0 1 0 1 0 1 1 0 0 1 0 1
    # # 1 1 1 1 1 0 0 1 0 1 0 1 1 0 0 1 0 1
    N_samples, N_days, N_features = dataset.shape
    print(f"{N_samples=}, {N_days=}, {N_features=}")
    args = get_train_args()
    
    # Split the data
    train_data, val_data, test_data = split_data(dataset)    
    print(f"{train_data.shape=}, {val_data.shape=}, {test_data.shape=}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #criterion
    criterion = nn.MSELoss()
    
    optimizer_type = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD # should expand to support more optimizers
    
    #grid search
    # Implement Optuna
    # Optuna (train_data, val_data, criterion, optimizer_type, device)
    batch_size = args.batch_size
    hidden_size = N_features // 2 if args.hidden_size == 'half' else int(args.hidden_size)
    lr = args.lr
    gradient_clip = args.grad_clip
    epochs = args.epochs
    
    if args.auto_regressor:
        X_train, y_train = create_autoregressive_data(train_data)
        train_data = TensorDataset(X_train, y_train)
        
        X_val, y_val = create_autoregressive_data(val_data)
        val_data = TensorDataset(X_val, y_val)
        
        X_test, y_test = create_autoregressive_data(test_data)
        test_data = TensorDataset(X_test, y_test)
        
    #create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    test_loader1 = DataLoader(test_data, batch_size=1)

    #create model
    if args.auto_regressor:
        model = LSTM_AR(N_features, hidden_size, bidirectional=args.bidirectional)
    else:
        model = LSTM_AE(N_features, hidden_size, bidirectional=args.bidirectional)
    print(model)
    
   
    # Make sure correct plot folder exits
    plot_folder = "./plots/SP500"
    os.makedirs(plot_folder, exist_ok=True)

    session_type = 'ae' if not args.auto_regressor else 'ar'
    #train
    plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      gradient_clip,
                      lr,
                      optimizer_type,
                      plot_folder,
                      session_type,
                      device,
                      )
    
    #evaluate
    test_loss, _ = evaluate(test_loader, model, criterion, session_type, device)
    print(f"Test loss: {test_loss}")
    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    
    number_of_plots = 3
    i = 0
    #plot some outputs pairs
    
        
    for data in test_loader1:
        if i == number_of_plots:
            break
        if args.auto_regressor:
            data, target = data
        output = model(data.to(device))
        if args.auto_regressor:
            x_hat, y_hat = output
        print(f"{type(output)=}, {type(data)=}")
        
        open, high, low, close = data.squeeze().permute(1, 0).detach().cpu().numpy()
        pred_open, pred_high, pred_low, pred_close = output.squeeze().permute(1, 0).detach().cpu().numpy()
        plt.subplot(2, 2, 1)
        plt.plot(open, label="Open")
        plt.plot(pred_open, label="Pred Open")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(high, label="High")
        plt.plot(pred_high, label="Pred High")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(low, label="Low")
        plt.plot(pred_low, label="Pred Low")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(close, label="Close")
        plt.plot(pred_close, label="Pred Close")
        plt.legend()
        plt.show()
        i += 1
        
if __name__ == "__main__":
    main()
        