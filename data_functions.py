import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler, SequentialSampler
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from some_func import get_time_of_day, holidays

# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def get_loaders(batch_size: int, data_train, validation_split=.2):
    """
    Возвращает кортеж объектов DataLoader, первый элемент - тренировочная выборка, второй - валидационная.
    Данные разделяются случайно
    DataLoader-ы будут передавать данные батчами для тренировки.

    :param batch_size: размер батча
    :param data_train: Данные, объект Dataset из Pytorch
    :param validation_split: доля данных, отводимых на валидацию
    :return: кортеж объектов DataLoader, первый элемент - тренировочная выборка, второй - валидационная.
    """

    # Определяем количество примеров в фолде валидации
    split = int(np.floor(validation_split * len(data_train)))

    # Список индексов для тренировочных примеров
    indices = list(range(len(data_train)))

    # Рандомизируем положение индексов в списке
    np.random.shuffle(indices)

    # Определяем списки с индексами примеров для тренировки и для валидации
    train_indices, val_indices = indices[split:], indices[:split]

    # Создаем семплеры, которые будут случайно извлекать данные из набора данных
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Создаем объекты типа ДатаЛоадер, которые будут передавать батчами данные в модель
    train_loader = DataLoader(data_train, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(data_train, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    return train_loader, val_loader


def get_alternative_loaders(batch_size: int, data_train, validation_split=.2):
    """
    Возвращает кортеж объектов DataLoader, первый элемент - тренировочная выборка, второй - валидационная.
    Данные разделяются последовательно
    DataLoader-ы будут передавать данные батчами для тренировки.

    :param batch_size: размер батча
    :param data_train: Данные, объект Dataset из Pytorch
    :param validation_split: доля данных, отводимых на валидацию
    :return: кортеж объектов DataLoader, первый элемент - тренировочная выборка, второй - валидационная
    """

    split = int(np.floor(validation_split * len(data_train)))
    indices = list(range(len(data_train)))
    train_indices, val_indices =  indices[split:], indices[:split]

    train_sampler = SequentialSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)

    train_loader = DataLoader(data_train, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(data_train, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    return train_loader, val_loader


class TurnstileDataset(Dataset):
    """
    Содержит в себе тренировочные и тестовые данные датасета с турникетами.

    """

    def __init__(self, test=False, normalize=False):
        """

        :param test: Используется для того, чтобы датасет возвращал значения тестовой выборки (когда используется, как итерируемый объект)
        :param normalize: Возможность включить\отключить нормализацию данных
        """
        self.train_data = pd.read_csv(os.path.join('train.csv'))
        self.test_data = pd.read_csv(os.path.join('test.csv'))

        self.X_train, self.y_train = get_train_test_data()[0]
        self.X_test, self.y_test = get_train_test_data()[1]

        self.normalize = normalize
        if self.normalize:
            self.X_train, self.X_test = self.normalize_data(self.X_train, self.X_test)

        self.test = test

    def __len__(self):
        if self.test:
            return self.X_test.shape[0]
        else:
            return self.X_train.shape[0]

    def __getitem__(self, idx):
        if self.test:
            X = torch.Tensor(self.X_test[idx, :])
            y = self.y_test[idx]
            return X, y
        else:
            X = torch.Tensor(self.X_train[idx, :])
            y = self.y_train[idx]
            return X, y

    @staticmethod
    def normalize_data(train_vector: np.ndarray, test_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Нормализует данные в каждом столбце, используя среднее значение и среднеквадратичное отклонение.

        :param train_vector: Тренировочные данные
        :param test_vector: Тестовые данные
        :return: Нормализованные тренировочные и тестовые данные (кортеж)
        """

        train_arr, test_arr = train_vector, test_vector
        united_arr = np.concatenate((train_arr, test_arr))

        mean = np.mean(united_arr, axis=0)
        std_deviation = np.std(united_arr, axis=0)

        train_X = (train_arr - mean) / std_deviation
        test_X = (test_arr - mean) / std_deviation

        return train_X, test_X

    @staticmethod
    def vectorize_data(filename: str = '1_clear_train.csv', test=False):
        """
        Чистит данные от ненужных столбцов, в том числе для синхронизации признаков в тренировочных и тестовых данных.
        Помимо чистки от ненужных столбцов, функция выполняет синхронизацию тренировочных и тестовых данных.
        По-хорошему, нужно было добавить несуществующие столбцы к нужным данным и заполнить их нулями, но из-аз чело-
        веческой лени в этой функции просто удаляются некоторые столбцы. Но так как их количество не велико, это не
        окажет значительного влияния на модель.


        :param filename: Путь к файлу
        :param test: Для переключения функции на обработку тестовых данных
        :return: Кортеж массивов NumPy
        """

        data = pd.read_csv(filename)

        to_drop = ['row_id', 'day_of_month', 'hour', 'min']  # Задаётся в ручную

        if not test:
            y_data_train = data.copy().user_id
            X_data_train = data.drop(columns=['user_id', 'gate_id_0', 'season_spring', 'hour_6'] + to_drop)
            X_data_train.to_csv('train1111.csv')
        else:
            X_data_train = data.drop(
                columns=['gate_id_2', 'season_spring', 'hour_0', 'hour_1', 'hour_3'] + to_drop)
            y_data_train = X_data_train.holiday
            X_data_train.to_csv('train2222.csv')

        return np.array(X_data_train.astype(int, copy=True)), np.array(y_data_train.astype(int, copy=True))


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # Преобразуем временную метку в объект datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Создаем признаки день недели, время суток, праздничный ли день
    data['day_of_the_week'] = data['timestamp'].dt.dayofweek
    data['part_of_the_day'] = data['timestamp'].dt.hour.apply(get_time_of_day)
    data['holiday'] = data['timestamp'].dt.date.astype(str).isin(holidays)
    data['holiday'] = data['holiday'].astype(int)

    # Создаем признак время года
    data['season'] = pd.cut(data['timestamp'].dt.month,
                            bins=[0, 3, 6, 9, 12],
                            labels=['winter', 'spring', 'summer', 'autumn'])

    # День месяца, часы, минуты, секунды
    data['day_of_month'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['min'] = data['timestamp'].dt.minute
    data['sec'] = data['timestamp'].dt.second

    # Время между проходами через конкретный турникет и время между каждой записью
    data['time_between_passes'] = data.groupby('gate_id')['timestamp'].diff().dt.total_seconds()
    data['time_between_passes'] = data['time_between_passes'].fillna(0)
    data['delta'] = data['timestamp'].diff().dt.total_seconds()
    data['delta'] = data['delta'].fillna(0)

    # Чистим
    data = data.drop(columns=['timestamp', 'row_id', ])

    return data


def get_train_test_data():
    data_test = pd.read_csv('test.csv')
    data_train = pd.read_csv('train.csv').drop(columns='user_id')

    data_test = get_features(data_test)
    data_train = get_features(data_train)
    #print(data_test.shape, data_train.shape)

    ohe = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                                   ['day_of_month', 'hour', 'min', 'gate_id', 'day_of_the_week', 'part_of_the_day',
                                    'season']),
                                  remainder='passthrough')

    vectorized_train = ohe.fit_transform(data_train)
    vectorized_test = ohe.transform(data_test)
    #print(vectorized_test.shape, vectorized_train.shape)

    y_train = np.array(pd.read_csv('train.csv').user_id, dtype=int)

    return (vectorized_train, y_train), (vectorized_test, vectorized_test[:, 0].astype(int))
