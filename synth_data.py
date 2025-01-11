import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def multiply_around_i(x):
    i = random.randint(20, 30)
    x[i-5:i+5] *= 0.1
    return x

def create_dataset(n_samples):
    X = np.random.rand(n_samples, 50)
    X = np.array([multiply_around_i(x) for x in X])
    return X

def split_dataset(X, y, val_size=0.2, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size))
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def plot_data(X, number_of_plots=5):
    random_selection = np.random.rand(number_of_plots)*len(X)
    random_selection = random_selection.astype(int)
    for i in random_selection:
        plt.plot(X[i], label= f"Index: {i}")
        
    plt.title("Synthetic data at {} random indices".format(number_of_plots))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("./plots/synthetic_data.png")
    plt.show()

if __name__ == "__main__":
    X = create_dataset(10000)
    print(f"len x: {len(X)}, len x[0]: {len(X[0])}")
    X_train, X_val, X_test, _,_,_ = split_dataset(X, X)
    print(f"shape x_train: {X_train.shape}, shape x_val: {X_val.shape}, shape x_test: {X_test.shape}")
    
    #save the data
    synth_path = "./synth_data"
    os.makedirs(synth_path, exist_ok=True)
    np.save(f"{synth_path}/X_train.npy", X_train)
    np.save(f"{synth_path}/X_val.npy", X_val)
    np.save(f"{synth_path}/X_test.npy", X_test)
    
    print(f"Saved all data to synth_data folder: {synth_path}")
    
    plot_data(X_train, 3)