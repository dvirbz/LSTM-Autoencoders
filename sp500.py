import os
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_daily_max(df, stocks):
    plt.figure(figsize=(12, 8))
    for stock in stocks:
        plt.subplot(len(stocks), 1, stocks.index(stock) + 1)
        sns.lineplot(data=df[df["symbol"] == stock], x="date", y="high")
    plt.show()    

def main():
    data_path = "./data/"
    file_name = "sp500.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    # print(df.head(), df.describe())
    # print(df.info())
    
    
    # plot_daily_max(df, ["AMZN", "GOOGL"])
    
    # Dataset Shape - (Batches, Sequence Length, Features) - (B, Amount of Days, 4)
    # Features: Open, High, Low, Close
    # Targets: Open, High, Low, Close
    
    # print(df[df.isna().any(axis=1)])
    # Clean Data
    df.dropna(inplace=True)
    df.drop(columns=["volume"], inplace=True)
    # Drop stocks with less than the maximum amount of days
    stock_days = df.groupby("symbol")["date"].count()
    max_days = stock_days.max()
    stocks_to_remove = stock_days[stock_days < max_days].index
    df = df[~df["symbol"].isin(stocks_to_remove)]

    # Now we know the shape we want is (B, 1007, 4)    
    dataset = df.pivot(index="date", columns="symbol", values=["open", "high", "low", "close"])
    print(dataset.shape)
    
if __name__ == "__main__":
    main()
        