import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.display.width = 0

def train_test_split(X, y, split_size=(0.8, 0.2)):
    cols = X.columns
    X, y = np.array(X), np.array(y)
    train_size = split_size[0]
    n_samples_train = int(y.size * train_size)

    new_indices = np.random.permutation(y.size)

    train_indices = new_indices[:n_samples_train]
    test_indices = new_indices[n_samples_train:]

    X_train = pd.DataFrame(X[train_indices], columns=cols)
    y_train = pd.Series(y[train_indices])
    X_test = pd.DataFrame(X[test_indices], columns=cols)
    y_test = pd.Series(y[test_indices])

    return X_train, y_train, X_test, y_test


