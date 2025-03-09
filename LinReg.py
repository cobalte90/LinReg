import pandas as pd
import numpy as np
import random

class LinReg:
    def __init__(self, X, y, learn_method, epochs):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.learn_method = learn_method
        self.w = np.array([random.randint(-100, 100) / 100 for x in range(len(X.columns))])  # Инициализация случайных весов
        self.b = random.randint(-100, 100) / 100  # Случайное смещение

    def func(self, X, w, b):  # Вычисление таргетов по фичам, весам и смещению
        return X.dot(w) + b

    def MSE(self):
        y_pred = self.func(self.X, self.w, self.b)
        mse = np.mean((self.y - y_pred) ** 2)
        return mse

    def fit(self, learning_rate):
        X = self.X # Фичи
        y = self.y # Таргет
        epochs = self.epochs # Количество циклов обучения


        for epoch in range(epochs): # Корректировка весов

            y_true = np.array(y)
            y_pred = self.func(X, self.w, self.b)
            self.b -= learning_rate * (2 / len(X)) * np.sum(y_pred - y_true)

            for i in range(len(self.w)):
                self.w[i] -= learning_rate * (2 / len(X)) * np.sum((y_pred - y_true) * X.iloc[:, i])


        return self.func(X, self.w, self.b)


    def predict(self, X_test):
        return self.func(X_test, self.w, self.b)
