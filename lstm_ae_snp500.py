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
from train_utils import evaluate, plot_train_losses, optuna_train

def plot_daily_max(df, stocks, save_path):
    plt.figure(figsize=(12, 10))
    os.makedirs(save_path, exist_ok=True)
    for stock in stocks:
        plt.subplot(len(stocks), 1, stocks.index(stock) + 1)
        ax = sns.lineplot(data=df[df["symbol"] == stock], x="date", y="high")
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("High", fontsize=14)
        plt.title(f"{stock} Daily High", fontsize=16)
    plt.savefig(f'{save_path}/daily_max.png')
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

    train_mean, train_std = train_data.mean(dim=0, keepdims=True), train_data.std(dim=0, keepdims=True)
    first_transform = transforms.Lambda(lambda x: torch.tensor(x,dtype=torch.float32))

    transform = transforms.Compose([
        first_transform,
        transforms.Normalize(mean=train_mean, std=train_std)
        ])

    train_data = transform(train_data)
    print(f"{train_data[:, 0, :]=}")
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

def plot_ar(test_loader, model, number_of_plots=3, device="cuda", save_path="./plots/SP500"):
    i = 0
    for data, _ in test_loader:
        if i == number_of_plots:
            break
        data = data.permute(1, 0, 2).to(device)
        first_data, second_data = train_test_split(data, test_size=0.5, shuffle=False)
        first_data = first_data.permute(1, 0, 2).to(device)
        first_data, second_data = train_test_split(data, test_size=0.5, shuffle=False)
        data = data.permute(1, 0, 2)
        first_data = first_data.to(device).permute(1, 0, 2)
        second_data = second_data.to(device).permute(1, 0, 2)
        N = second_data.shape[1]
        input_seq = data[:, :N, :].to(device)
        second_data_hat = torch.zeros(second_data.shape[0], N, second_data.shape[2]).to(device)
        for j in range(N):
            y_hat = model.generate(input_seq).to(device)
            second_data_hat[:, j, :] = y_hat
            input_seq = data[:, j:j+N, :]

        one_step_preds = model(data)[1].to(device)
        shift = 1
        one_step_preds = torch.cat((data[:, 0:shift, :], one_step_preds[:, :-shift, :]), dim=1)
        
        pred_data = torch.cat((first_data, second_data_hat), dim=1)
        open, high, low, close = data.squeeze().permute(1, 0).detach().cpu().numpy()
        pred_open, pred_high, pred_low, pred_close = pred_data.squeeze().permute(1, 0).detach().cpu().numpy()
        one_step_open, one_step_high, one_step_low, one_step_close = one_step_preds.squeeze().permute(1, 0).detach().cpu().numpy()
        
        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(open, label="Open")
        plt.plot(pred_open, label="Pred Open")
        plt.plot(one_step_open, label="One Step Pred Open")
        plt.axvline(first_data.shape[1], color='r', linestyle='--')
        plt.title("Open")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(high, label="High")
        plt.plot(pred_high, label="Pred High")
        plt.plot(one_step_high, label="One Step Pred High")
        plt.axvline(first_data.shape[1], color='r', linestyle='--')
        plt.title("High")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(low, label="Low")
        plt.plot(pred_low, label="Pred Low")
        plt.plot(one_step_low, label="One Step Pred Low")
        plt.axvline(first_data.shape[1], color='r', linestyle='--')
        plt.title("Low")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(close, label="Close")
        plt.plot(pred_close, label="Pred Close")
        plt.plot(one_step_close, label="One Step Pred Close")
        plt.axvline(first_data.shape[1], color='r', linestyle='--')
        plt.title("Close")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        
        plt.savefig(f"{save_path}/AR_plot_{i}.png")
        plt.show()
        i += 1

        
def main():
    data_path = "./data/"
    file_name = "sp500.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # plot_daily_max(df, ["AMZN", "GOOGL"], 'plots/SP500')
    
    dataset = preprocess_data(df)

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
    model_type = args.model_type
    best_params = None

    batch_size = args.batch_size
    hidden_size = N_features // 2 if args.hidden_size == 'half' else int(args.hidden_size)
    lr = args.lr
    gradient_clip = args.grad_clip
    epochs = args.epochs if not args.hyper_search else args.trial_epochs

    if model_type == "ar":
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

    if args.hyper_search:
        best_params = optuna_train(train_loader,
                                   val_loader,
                                   criterion,
                                   optimizer_type,
                                   args.trial_epochs,
                                   model_type,
                                   args.n_trials,
                                   hidden_size,
                                   device
                                   )
        print(f"Best params: {best_params}")
        hidden_size = best_params["hidden_size"]
        lr = best_params["lr"]
        gradient_clip = best_params["grad_clip"]

        
    #create model
    if model_type == 'ar':
        model = LSTM_AR(N_features, hidden_size, bidirectional=args.bidirectional)
    else:
        model = LSTM_AE(N_features, hidden_size, bidirectional=args.bidirectional)
    print(model)
    
   
    # Make sure correct plot folder exits
    plot_folder = "./plots/SP500"
    os.makedirs(plot_folder, exist_ok=True)

    #train
    plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      gradient_clip,
                      lr,
                      optimizer_type,
                      plot_folder,
                      model_type,
                      device,
                      )
    
    #evaluate
    if model_type == "ar":
        encoder_loss, regressor_loss = evaluate(test_loader, model, criterion, model_type, device)
        test_loss = encoder_loss + regressor_loss
    else:
        test_loss, _ = evaluate(test_loader, model, criterion, model_type, device)
    print(f"Test loss: {test_loss}")
    #save model
    models_folder = "./models"
    os.makedirs(models_folder, exist_ok=True)
    model_name = f"model_{hidden_size=}_{lr=}_{gradient_clip=}_{epochs=}_{batch_size=}_{test_loss=}"
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
    
    number_of_plots = 3
    if model_type == "ae":
        plot_ae(test_loader1, model, number_of_plots)
    else:
        plot_ar(test_loader1, model, number_of_plots)
        
if __name__ == "__main__":
    main()
        