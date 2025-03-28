import pandas as pd
import numpy as np
import random


class LinReg:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None

    def fit(self, learn_method='gradient', learning_rate=0.01, epochs=1000):

        # Добавляем колонку с единицами для смещения
        X = np.c_[np.ones(self.X.shape[0]), self.X]  # Добавляем колонку единиц

        self.w = np.random.rand(X.shape[1])  # Инициализация весов

        if learn_method == 'gradient':
            for epoch in range(epochs):
                y_true = np.array(self.y)
                y_pred = self._lin_model(X, self.w)

                error = y_pred - y_true
                grad = (2 / len(X)) * np.dot(X.T, error)
                self.w -= learning_rate * grad

        elif learn_method == 'analytical':
            X_transposed = X.T
            features_corr_matrix = X_transposed.dot(X)
            inverse_matrix = np.linalg.inv(features_corr_matrix)
            new_w = inverse_matrix.dot(X_transposed).dot(self.y)

            self.w = new_w

    def predict(self, X_test):
        # Добавляем колонку с единицами для смещения
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return self._lin_model(X_test, self.w)

    def _lin_model(self, X, w):
        return X.dot(w)
