# Решение конкруса "Турникеты"

Решение основано на нейронной сети с линейными слоями, реализовано при помощи фреймворка pyTorch. Точнсть на тестовой выборке 0.113

* Первичный анализ - файл `Feature_Engineering.ipynb`
* Создание признаков и формирование конечного датасета для обучения - файл `data_functions.py`
* Архитектура модели и её обучение - файл `main.py`

Подбор гиперпараметров реализован с помощью Optuna.
