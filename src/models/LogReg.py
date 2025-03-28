import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Логистическая регрессия
class LogReg:
    def __init__(self):
        self.w = None
        self.b = 0

    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))

    def linear_model(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y, epochs=100, learning_rate=0.01):

        # Инициализация весов
        self.w = np.random.rand(X.shape[1])

        for eph in range(epochs):
            # Предсказание целевой переменной
            y_pred = self.sigmoid(self.linear_model(X))

            # Градиенты по весам и смещению
            grad_w = - (1 / X.shape[0]) * np.dot(X.T, (y - y_pred))
            grad_b = - (1 / X.shape[0]) * np.sum(y - y_pred)

            # Обновление весов и смещения
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b

        return "Модель обучена!"

    def predict(self, X):
        y_pred = self.sigmoid(self.linear_model(X))
        return np.array([ 1 if i >= 0.5 else 0 for i in y_pred])

