import pandas as pd
import numpy as np
import random

class LinReg:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.array([random.randint(-100, 100) / 100 for _ in range(len(X.columns))])  # Инициализация случайных весов
        self.b = random.randint(-100, 100) / 100  # Случайное смещение

    def _lin_model(self, X, w, b):  # Вычисление таргетов по фичам, весам и смещению
        return X.dot(w) + b


    def fit(self, learn_method='gradient' , learning_rate=0.05, epochs=1000):
        X = self.X # Фичи
        y = self.y # Таргет
        w = self.w # Веса
        b = self.b # Смещение

        if learn_method == 'gradient':
            for epoch in range(epochs): # Корректировка весов

                y_true = np.array(y)
                y_pred = self._lin_model(X, w, b)
                b -= learning_rate * (2 / len(X)) * np.sum(y_pred - y_true)

                error = y_pred - y_true
                grad = (2 / len(X)) * np.dot(X.T, error)
                w -= learning_rate * grad

            self.w = w
            self.b = b

        if learn_method == 'analytical':

            X['ones_for_bias'] = [1 for _ in range(len(X))] # Добавляем колонку с единицами для смещения
            X_transposed = X.T
            features_corr_matrix = X_transposed.dot(X)
            inverse_matrix = np.linalg.inv(features_corr_matrix)
            new_w = inverse_matrix.dot(X_transposed).dot(y)

            self.w = new_w


    def predict(self, X_test):
        return self._lin_model(X_test, self.w, self.b)


def MAE(y_true, y_pred):
    mae = np.mean(abs(y_true - y_pred))
    return mae