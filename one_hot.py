# Кодирование категориальных переменных методом one-hot
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
pd.options.display.width = 0

'''
Функция для one-hot-encoding
Параметры: X - датасет, columns - список индексов колонок для кодирования
'''
def one_hot(X, columns=[0]):
    '''
        Перебираем индексы колонок в датасете.
            Если колонку нужно кодировать:
                Создаем пустой словарь.
                Берем список уникальных значений в колонке.
                Для каждого уникального значения создаём список из нулей и единиц.
                Передаем этот список в словарь по ключу: (имя исходной колонки)_(номер уникального значения)
            Добавляем словарь новых колонок в датасет.

        Помещаем исходные колонки в список columns_to_drop.
        Удаляем из датасета исходные колонки.
        Возвращаем датасет.

    '''

    for col_index in range(len(X.columns)):

        if col_index in columns:
            col_name = X.columns[col_index]
            values = X[col_name].unique()
            d = {} # Словарь для новых колонок
            for i in range(len(values)):
                temp_key = col_name+'_'+str(i)
                d[temp_key] = [1 if x == values[i] else 0 for x in X[col_name]]
            for col in d:
                X[col] = d[col]

    columns_to_drop = []
    for i in range(len(X.columns)):
        if i in columns:
            columns_to_drop.append(X.columns[i])
    X = X.drop(columns_to_drop, axis=1)
    return X
