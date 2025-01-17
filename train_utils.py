import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import optuna

from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from lstm_AE import LSTM_AE, LSTM_AE_Classifier, LSTM_AR


class SqrtSched(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        return [base_lr / np.sqrt(self.last_epoch + 1) for base_lr in self.base_lrs]

def train_epoch_AE(train_loader, model, optimizer, criterion, grad_clip, device):
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.float().to(device)
        output = model(data).to(device)
        loss = criterion(output, data)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()

    total_loss = running_loss / len(train_loader)
    return total_loss

def train_epoch_CLS(train_loader, model, optimizer, criterion, grad_clip, device):
    running_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    for data, targets in train_loader:
        optimizer.zero_grad()
        targets = one_hot(targets, 10).to(torch.float32).to(device) # fix to generalize            
        data = data.float().to(device)
        output, probs = model(data)
        output = output.to(device)
        probs = probs.to(device)
        ce_loss = ce_criterion(probs, targets)
        ae_loss = criterion(output, data)
        loss = ae_loss + ce_loss
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()

    total_loss = running_loss / len(train_loader)
    return total_loss

def train_epoch_regressor(train_loader, model, optimizer, criterion, grad_clip, device, lambda_ar=1):
    running_loss = 0.0
    ar_running_loss = 0.0
    ae_running_loss = 0.0
    for data, targets in train_loader:
        optimizer.zero_grad()
        targets = targets.to(device)
        data = data.float().to(device)
        x_hat, y_hat = model(data)
        ar_loss = lambda_ar * criterion(y_hat, targets)
        ae_loss = criterion(x_hat, data)
        loss = ar_loss + ae_loss
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()
        ar_running_loss += ar_loss.item()
        ae_running_loss += ae_loss.item()

    total_loss = running_loss / len(train_loader)
    ar_total_loss = ar_running_loss / len(train_loader)
    ae_total_loss = ae_running_loss / len(train_loader)
    return total_loss, ar_running_loss, ae_running_loss

def evaluate_AE(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.float().to(device)
            output = model(data).to(device)
            all_outputs.append((data, output))
            loss = criterion(output, data)
            accumulative_loss += loss.item()

    total_loss = accumulative_loss / len(val_loader)
    return total_loss, all_outputs

def evaluate_CLS(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for data, targets in val_loader:
            labels.extend(targets.to(device))
            targets = one_hot(targets, 10).to(torch.float32).to(device) # fix to generalize
            data = data.float().to(device)
            
            output, probs = model(data)          
            output = output.to(device)
            probs = probs.to(device)
            
            curr_preds = torch.argmax(probs, dim=1)
            preds.extend(curr_preds)
            
            ce_loss = ce_criterion(probs, targets)
            ae_loss = criterion(output, data)
            loss = ce_loss + ae_loss
            accumulative_loss += loss.item()

    accuracy = (torch.tensor(labels) == torch.tensor(preds)).sum().item() / len(labels)
    total_loss = accumulative_loss / len(val_loader)
    return total_loss, accuracy

def evaluate_regressor(val_loader, model, criterion, device):
    accumulative_loss = 0.0
    ar_accumulative_loss = 0.0
    ae_accumulative_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data, targets in val_loader:
            targets = targets.to(device)
            data = data.float().to(device).permute(1, 0, 2)
            _, second_data = train_test_split(data, test_size=0.5, shuffle=False)
            data = data.permute(1, 0, 2)
            second_data = second_data.to(device).permute(1, 0, 2)
            N = second_data.shape[1]
            input_seq = data[:, : N, :].to(device)
            second_data_hat = torch.zeros(second_data.shape[0], N, second_data.shape[2]).to(device)
            for i in range(N):
                y_hat = model.generate(input_seq).to(device)
                second_data_hat[:, i, :] = y_hat
                input_seq = data[:, i : i + N, :]

            # second_data_hat = model.generate(first_data, second_data.shape[1]).to(device)
            ar_loss = criterion(second_data, second_data_hat)
            ae_loss = criterion(data, model(data)[0])
            loss = ar_loss + ae_loss
            accumulative_loss += loss.item()
            ar_accumulative_loss += ar_loss.item()
            ae_accumulative_loss += ae_loss.item()

    total_loss = accumulative_loss / len(val_loader)
    ar_total_loss = ar_accumulative_loss / len(val_loader)
    ae_total_loss = ae_accumulative_loss / len(val_loader)
    return total_loss, ar_total_loss, ae_total_loss

MODELS = {
    'ae' : {
            'MODEL' : LSTM_AE,
            'TRAINER' : train_epoch_AE,
            'EVALUATOR' : evaluate_AE
            },
    'cls' : {
            'MODEL' : LSTM_AE_Classifier,
             'TRAINER' : train_epoch_CLS,
             'EVALUATOR' : evaluate_CLS
            },
    'ar' : {
            'MODEL' : LSTM_AR,
            'TRAINER' : train_epoch_regressor,
            'EVALUATOR' : evaluate_regressor
            },
}


def evaluate(val_loader, model, criterion, model_type, device):
    evaluator = MODELS[model_type]['EVALUATOR']
    return evaluator(val_loader, model, criterion, device)

def train(train_loader,
          model,
          criterion,
          epochs,
          grad_clip,
          learning_rate,
          optimizer_type,
          model_type,
          device,
          ):
    model.to(device)
    model.train()
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in tqdm(range(epochs), desc="Training", leave=False):
        trainer = MODELS[model_type]['TRAINER']
        train_loss = trainer(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        # tqdm.write(f"Epoch: {epoch}, Loss: {train_loss}")
    
    return train_loss

def plot_train_losses(train_loader,
                      model,
                      criterion,
                      epochs,
                      grad_clip,
                      learning_rate,
                      optimizer_type,
                      save_path,
                      model_type,
                      device,
                      with_accuracy=False,
                      ):
    """
    Train a neural network model and display an interactive graph of training
    and validation losses.

    Args:
        model (NeuralNetwork): The neural network model to train.
        loss_fn (Loss): The loss function used for training.
        optim (Optimizer): The optimizer used for parameter updates.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        X_train (np.ndarray): Training data features.
        C_train (np.ndarray): Training data labels.
        X_val (np.ndarray): Validation data features.
        C_val (np.ndarray): Validation data labels.
    """
    # Initialize variables to store losses
    train_losses = []
    train_accs = []
    # Prepare the interactive plot
    if with_accuracy:
        _, ax = plt.subplots(2, figsize=(9, 9))
        axis = ax
    else:
        _, ax = plt.subplots()
        axis = [ax]
    for a in axis:
        if a == axis[0]:
            a.set_xlabel('Epoch')
            a.set_ylabel('Loss')
            a.set_title('Training Loss')
            train_line, = a.plot([], [], label='Train Loss', color='blue')
        else:
            a.set_ylabel('Accuracy')
            a.set_title('Training Accuracy')
            acc_line, = a.plot([], [], label='Train Accuracy', color='blue')

    # Add a text box for learning rate
    for a in axis:
        if a == axis[0]:
            lr_text = a.text(0.1, 0.95, '', transform=a.transAxes)
            loss_text = a.text(0.1, 0.85, '', transform=a.transAxes)
        else:
            acc_text = a.text(0.1, 0.95, '', transform=a.transAxes)

    # Helper function to update the graph
    def update_graph(train_loss, train_acc, learning_rate):
        train_losses.append(train_loss)
        if with_accuracy:
            train_accs.append(train_acc)
        train_line.set_data(range(1, len(train_losses) + 1), train_losses)
        if with_accuracy:
            acc_line.set_data(range(1, len(train_accs) + 1), train_accs)
        for a in axis:
            a.relim()
            a.autoscale_view()
        loss_text.set_text(f'Train Loss: {train_loss:.4f}')
        if with_accuracy:
            acc_text.set_text(f'Train Accuracy: {train_acc:.4f}')
        lr_text.set_text(f'Learning Rate: {learning_rate:.5e}')
        plt.draw()
        plt.pause(0.01)

    model.to(device)
    model.train()
    train_acc = None

    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    print(f"Optimizer: {optimizer}")
    scheduler = SqrtSched(optimizer)
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        trainer = MODELS[model_type]['TRAINER']
        train_loss = trainer(train_loader, model, optimizer, criterion, grad_clip, device)
        losses.append(train_loss)
        if with_accuracy:
            # data = train_loader.dataset.data.to(device).to(torch.float32)
            # pred = torch.argmax(model(data)[1], dim=1)
            # targets = train_loader.dataset.targets.to(device)
            # tqdm.write(f"{pred=}, {targets=}")
            # train_acc = (pred == targets).to(torch.float32).mean().item()
            train_acc = evaluate(train_loader, model, criterion, model_type, device)[1]
            model.train()
        update_graph(train_loss, train_acc, optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.savefig(f'{save_path}_loss_plots.png')
    plt.show()

    print(f"Training complete.\nModel final loss: {train_losses[-1]:.2f}")


def optuna_train(
  train_loader,
  val_loader,
  criterion,
  optimizer_type,
  n_epochs,
  model_type,
  n_trials,
  hidden_size_limit,
  device,
):
    def objective(trial):
        hyperparams = {}
        hyperparams["lr"] = trial.suggest_categorical("lr", np.logspace(-3.5, -1.5, num=5))
        hyperparams["grad_clip"] = trial.suggest_categorical("grad_clip", np.arange(0.1, 2.1, 0.4))
        hyperparams["hidden_size"] = trial.suggest_int("hidden_size", 1, hidden_size_limit)
        if model_type == 'cls':
            input_shape = train_loader.dataset.data.shape[1]
        elif model_type == 'ae':
            input_shape = train_loader.dataset.shape[1]
        elif model_type == 'ar':
            input_shape = train_loader.dataset[0][0].shape[1]

        model = MODELS[model_type]['MODEL']
        if model_type == 'cls':
            model = model(input_shape, hyperparams["hidden_size"], len(np.unique(train_loader.dataset.targets)), bidirectional=False)
        else:
            model = model(input_shape, hyperparams["hidden_size"], bidirectional=False)
        model.to(device)
        train(train_loader,
              model,
              criterion,
              n_epochs,
              hyperparams["grad_clip"],
              hyperparams["lr"],
              optimizer_type,
              model_type,
              device
              )

        val_loss, accuracy = evaluate(val_loader, model, criterion, model_type, device)
        del model
        if model_type == 'cls':
            return 1 - accuracy
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params