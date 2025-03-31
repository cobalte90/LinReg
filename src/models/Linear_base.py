import numpy as np

class Linear():
    
    def predict(self, X_test):
        # Добавляем колонку с единицами для смещения
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return self._linear_model(X_test, self.w)

    
    def _linear_model(self, X, w):
        return X.dot(w)