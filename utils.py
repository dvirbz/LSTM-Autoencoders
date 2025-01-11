import numpy as np

def load_data(data_folder):
    X_train = np.load(f"{data_folder}/X_train.npy")
    X_val = np.load(f"{data_folder}/X_val.npy")
    X_test = np.load(f"{data_folder}/X_test.npy")
    return X_train, X_val, X_test