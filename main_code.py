import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from one_hot import one_hot
from LinReg import LinReg
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df = pd.read_excel('bike.xlsx')
X = df.drop('cnt', axis=1)
y = df['cnt']

# Кодирование категориальных фичей
X = one_hot(X)

# Обучение модели
model = LinReg(X, y, learn_method=1, epochs=1000) # learn_meathod пока не реализован
model.fit(learning_rate=0.01)
y_pred = model.predict(X)

# Обучение модели sklearn
model_sklearn = Lasso(alpha=10)
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)
plt.scatter(y_pred_sklearn, y)
sns.heatmap(X.corr(), cmap='coolwarm')
# plt.show()

print(
    f'Значение MSE у моей модели: {mse(y, y_pred)}\nЗначение MSE у модели sklearn: {mse(y, y_pred_sklearn)}'
)