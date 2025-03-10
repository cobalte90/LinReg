import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from one_hot import one_hot
from LinReg import *
from EDA import EDA

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
pd.set_option('display.max_columns', None)
pd.options.display.width = 0


df = pd.read_excel('bike.xlsx')

# Разведочный анализ, графики
EDA(df)

X = df.drop('cnt', axis=1)
y = df['cnt']

# Кодирование категориальных фичей
X = one_hot(X)

# Обучение модели
model = LinReg(X, y, learn_method=1, epochs=1000) # learn_meathod пока не реализован

for i in range(1, 10):
    model = LinReg(X, y, learn_method=1, epochs=1000)
    model.fit(learning_rate=i/100)
    print(MAE(y, model.predict(X)), i/100)

model.fit(learning_rate=0.1)
y_pred = model.predict(X)
print(MAE(y, y_pred), 0.1)

# Обучение модели sklearn
model_sklearn =LinearRegression()
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)


print(
    f'Значение MAE у моей модели: {MAE(y, y_pred)}\nЗначение MAE у модели sklearn: {mae(y, y_pred_sklearn)}'
)