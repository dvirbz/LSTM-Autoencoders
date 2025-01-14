import os
import torch
from torch import nn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from argparser import get_train_args
from lstm_AE import LSTM_AE
from train_utils import evaluate, plot_train_losses

def plot_daily_max(df, stocks):
    plt.figure(figsize=(12, 8))
    for stock in stocks:
        plt.subplot(len(stocks), 1, stocks.index(stock) + 1)
        sns.lineplot(data=df[df["symbol"] == stock], x="date", y="high")
    plt.show()    
    
def preProcessData(df):
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
    dataset = preProcessData(df)
    N_samples, N_days, N_features = dataset.shape
    print(f"{N_samples=}, {N_days=}, {N_features=}")
    
    # Split the data
    dataset = torch.tensor(dataset, dtype=torch.float32)
    dataset = dataset.permute(1, 0, 2)
    train_data, test_data = train_test_split(dataset.detach().cpu(), test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.25)
    
    print(f"{train_data.shape=}, {val_data.shape=}, {test_data.shape=}")
    
    args = get_train_args()
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
    
    #create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    #create model
    model = LSTM_AE(N_features, hidden_size, bidirectional=args.bidirectional)
    print(model)
    
   
    # Make sure correct plot folder exits
    plot_folder = "./plots/SP500"
    os.makedirs(plot_folder, exist_ok=True)

    #train
    train_loss = plot_train_losses(train_loader, model, criterion, epochs, gradient_clip, lr, optimizer_type, plot_folder, device)
    
    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    
    #evaluate
    test_loss, all_outputs = evaluate(test_loader, model, criterion, device)
    
    #plot some outputs pairs
    
if __name__ == "__main__":
    main()
        