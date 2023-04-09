import optuna
from optuna.trial import TrialState
import configparser
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data_functions import TurnstileDataset, get_loaders
from metrics_functions import compute_binary_accuracy, validation_loss
from datetime import datetime

CLASSES = 58

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def define_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []

    in_features = 34
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 34, 200)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU(inplace=True))
        p = trial.suggest_float('droput_l{}'.format(i), .0, .5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    softm = trial.suggest_int('have_softmax', 0, 1)
    if softm == 1:
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)


def objective(trial):
    model = define_model(trial)
    model.to(device)
    model.type(torch.cuda.FloatTensor)

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adagrad', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-2)
    factor = trial.suggest_float('LR_Annealing_factor', .05, .9)
    patience = trial.suggest_int('LR_Annealing_patience', 3, 50)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    num_epochs = trial.suggest_int('num_epochs', 200, 500)
    batch_size = trial.suggest_int('batch_size', 64, 512)
    validation_split = 0.1
    data_train = TurnstileDataset(normalize=True)
    train_loader, val_loader = get_loaders(batch_size=batch_size, data_train=data_train,
                                           validation_split=validation_split)

    config = configparser.ConfigParser()
    config.read('config.ini')

    params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'validation_split': validation_split,
        'optimizer': optimizer_name,
        'annealing_factor': factor,
        'annealing_patience': patience
    }

    loss_history = []
    val_loss_history = []
    train_history = []
    val_history = []
    lr_history = []

    for epoch in tqdm(range(num_epochs)):
        model.train()

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
            loss_value = loss(prediction, y)

            optimizer.zero_grad()
            loss_value.backward()
            # Обновляем веса
            optimizer.step()

            # Определяем индексы, соответствующие выбранным моделью лейблам
            _, indices = torch.max(prediction, dim=1)
            # Сравниваем с ground truth, сохраняем количество правильных ответов
            correct_samples += torch.sum(indices == y)
            # Сохраняем количество всех предсказаний
            total_samples += y.shape[0]
            # run['train/batch/acc'].append(correct_samples / total_samples)

            loss_accum += loss_value

        # Среднее значение функции потерь за эпоху
        ave_loss = loss_accum / (i_step + 1)
        # Рассчитываем точность тренировочных данных на эпохе
        train_accuracy = float(correct_samples) / total_samples
        # Рассчитываем точность на валидационной выборке (вообще после этого надо бы гиперпараметры поподбирать...)
        val_accuracy = compute_binary_accuracy(model, val_loader)

        # Сохраняем значения ф-ии потерь и точности для последующего анализа и построения графиков
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        # Посчитаем лосс на валидационной выборке
        val_loss = validation_loss(model, val_loader, loss)
        val_loss_history.append(val_loss)

        trial.report(val_loss, epoch)

        lr_history.append(scheduler.optimizer.param_groups[0]['lr'])
        # Уменьшаем лернинг рейт (annealing)
        scheduler.step(val_loss)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize", study_name='first_try_turnstiles',
                                storage=f'sqlite:///{datetime.now().strftime("optuna_%d%m%y%H%M")}.db')
    study.optimize(objective, n_trials=2400, timeout=46800)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
