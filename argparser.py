import argparse

#define training arguments
def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=str, default='half', help='Hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--grad_clip', type=float, default=1, help='Gradient clipping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--data_folder', type=str, default='synth_data/', help='Data folder')
    parser.add_argument('--bidirectional', action='store_true', help='whether to use bidirectional LSTM')
    parser.add_argument('--pixel_wise', action='store_true', help='whether to use pixel-wise transformation')
    parser.add_argument('--hyper_search', action='store_true', help='Whether to perform hyperparameter search')
    parser.add_argument('--n_trials', type=int, default=20, help='number of trials for hyperparameter search')
    parser.add_argument('--trial_epochs', type=int, default=100, help='Number of epochs per trial for hyperparameter search')
    parser.add_argument('--auto_regressor', action='store_true', help='Whether to use the auto-regressor model')

    return parser.parse_args()