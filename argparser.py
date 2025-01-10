import argparse

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=20, help='Hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--data_folder', type=str, default=None, help='Data folder')
    parser.add_argument('--grid_search', action='store_true', help='Grid search')
    return parser.parse_args()