"""
Обучение модели.
"""

import configparser
import os
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import neptune
from data_functions import TurnstileDataset, get_loaders, get_alternative_loaders
from metrics_functions import compute_accuracy, validation_loss

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

# Модель
model = nn.Sequential(
    nn.Linear(141, 58),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.069),
    nn.Linear(128, 139),
    nn.ReLU(),
    nn.Dropout(0.0155),
    nn.Linear(139, 195),
    nn.ReLU(),
    nn.Dropout(0.017),
    nn.Linear(195, 58),
)
model.to(device)
model.type(torch.cuda.FloatTensor)

# Hyper Params
num_epochs = 300
batch_size = 142
learning_rate = 0.00235
weight_decay = 0.0001
validation_split = .15

# Загрузка данных
data_train = TurnstileDataset(normalize=True)
train_loader, val_loader = get_alternative_loaders(batch_size=batch_size, data_train=data_train, validation_split=validation_split)

# Loss Function, Optimizer
loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# LR Annealing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.83, patience=7)

config = configparser.ConfigParser()
config.read('config.ini')

run = neptune.init_run(
    project="pas-zhukov/ODS-Turnstile",
    api_token=config['Config']['api_token'],
    source_files=['Investigation.ipynb', 'main.py', 'data_functions.py']
)
params = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'validation_split': validation_split,
    'optimizer': 'Adam',
    'annealing_factor': .83
}
run["parameters"] = params

loss_history = []
val_loss_history = []
train_history = []
val_history = []
lr_history = []

# Training
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
        loss_accum += loss_value

    # Среднее значение функции потерь за эпоху
    ave_loss = loss_accum / (i_step + 1)
    # Рассчитываем точность тренировочных данных на эпохе
    train_accuracy = float(correct_samples) / total_samples
    # Рассчитываем точность на валидационной выборке (вообще после этого надо бы гиперпараметры поподбирать...)
    val_accuracy = compute_accuracy(model, val_loader)

    # Сохраняем значения ф-ии потерь и точности для последующего анализа и построения графиков
    loss_history.append(float(ave_loss))
    train_history.append(train_accuracy)
    val_history.append(val_accuracy)

    # Посчитаем лосс на валидационной выборке
    val_loss = validation_loss(model, val_loader, loss)
    val_loss_history.append(val_loss)

    run['train/epoch/loss'].append(ave_loss)
    run['valid/epoch/loss'].append(val_loss)
    run['train/epoch/acc'].append(train_accuracy)
    run['valid/epoch/acc'].append(val_accuracy)
    run['train/epoch/lr'].append(scheduler.optimizer.param_groups[0]['lr'])

    lr_history.append(scheduler.optimizer.param_groups[0]['lr'])
    # Уменьшаем лернинг рейт (annealing)
    scheduler.step(val_loss)

print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))


# Сохраняем веса! (А вдруг...)
torch.save(
        model.state_dict(),
        os.path.join('optimized/optim' + datetime.datetime.now().strftime('%d%m%y%H%M') + '.pth')
    )

# Сразу сделаем файл с предсказаниями
test_set = TurnstileDataset(test=True, normalize=True)
model.eval()
predictions = model(torch.Tensor(test_set.X_test).to(device))

predictions_labels = torch.argmax(predictions, dim=1)
predictions_labels_list = list(np.array(predictions_labels.cpu()))

output = pd.DataFrame({'row_id' : list(range(37518, 44643)),
                       'target' : predictions_labels_list}, columns=['row_id', 'target'])
output.to_csv(os.path.join('outputs/output__' + datetime.datetime.now().strftime('%d%m%y%H%M') + '.csv'), index=False)
