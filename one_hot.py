# Кодирование категориальных переменных методом one-hot
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.display.width = 0

# one-hot-encoding
def encoding(X, columns): # X - DataFrame, columns - names of columns to encode

    for col_name in columns: # iterate over columns names
        current_column = X[col_name] # current column
        values = np.unique(current_column) # unique values in current column
        new_array = np.zeros((current_column.size, values.size)).astype(int) # 2-d array with size (current_column, unique_values)
        row_indices = np.arange(current_column.size) # indices of rows in new array
        col_indices = np.searchsorted(values, current_column) # indices of columns in new array
        new_array[row_indices, col_indices] = 1 # insert 1 into the appropriate cells
        new_df = pd.DataFrame(new_array, columns = values) # transform to DataFrame
        X = pd.concat([X, new_df], axis=1) # concatenation
        X = X.drop(col_name, axis=1) # delete original column
    return X

# if the parameter "columns" if not specified or '', do encoding for all columns with type "object"
def one_hot(X, columns=''):
    if columns == '':
        return encoding(X, [i for i in X.columns if X[i].dtype == 'object'])
    else:
        return encoding(X, columns)

