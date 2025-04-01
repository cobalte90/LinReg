import pandas as pd
import numpy as np
from src.preprocessing.class_weights import *

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


class LogRegMultiClass:
    def __init__(self):
        self.w = None
        self.is_bias = False
        self.class_weights = None

    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # Для стабильности
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # one vs all
    def fit_one_vs_all(self, X, y, epochs=1000, learning_rate=0.1):

        X, y = np.array(X), np.array(y)
        
        if self.is_bias == False:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            self.bias = True

        classes = np.unique(y)
        n_classes = classes.shape[0]

        # Wiegths init random
        self.w = np.random.rand(n_classes, X.shape[1])

        for current_class in range(n_classes):
            new_y = (y == current_class).astype(int)

            for eph in range(epochs):
                y_pred = self.sigmoid(np.dot(X, self.w[current_class]))
                error = y_pred - new_y

                self.w[current_class] -= learning_rate * (1 / X.shape[0]) * np.dot(X.T, error)
        
        return self.w
    
    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        w = self.w
        z = np.dot(X, w.T)
        probabilities = self.softmax(z)
        return np.argmax(probabilities, axis=1)
