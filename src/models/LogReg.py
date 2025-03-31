import pandas as pd
import numpy as np
from src.preprocessing.class_weights import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Логистическая регрессия
class LogReg:
    def __init__(self):
        self.w = None
        self.is_bias = False
        self.class_weights = None

    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))

    def linear_model(self, X):
        return np.dot(X, self.w)

    def fit(self, X, y, epochs=1000, learning_rate=0.5):
        if self.is_bias == False:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            self.bias = True
        # Инициализация весов
        self.w = np.random.rand(X.shape[1])

        # Взвешивание классов
        if self.class_weights == None:
            self.class_weights = compute_class_weight(y)
        weighted_classes = np.array([self.class_weights[i] for i in y])

        for eph in range(epochs):
            y_pred = self.sigmoid(X @ self.w)
            error = y_pred - y
            grad = np.dot(X.T, error) / X.shape[0]

            self.w -= learning_rate * grad

        return self

    def predict(self, X):
        if self.is_bias == False:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = self.sigmoid(X @ self.w)
        return np.where(y_pred >= 0.5, 1, 0)

