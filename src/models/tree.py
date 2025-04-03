import numpy as np
import pandas as pd
from collections import Counter

class Node:
    # tree Node
    def __init__(self, is_leaf: bool = False, feature_index: int=None, threshold: float=None, 
                 left=None, right=None, value: int=None, n_samples: int=None, gain: float=None):
        self.is_leaf = is_leaf # the last node in the branch
        self.feature_index = feature_index # feature to split
        self.threshold = threshold # threshold to split
        self.left = left # left samples
        self.right = right # right samples
        self.value = value # class label
        self.n_samples = n_samples # number of samples input
        self.gain = gain # information gain in current node
 
class DecisionTreeClassifier:
    def __init__(self, max_depth: int=4, min_samples_split: int=5, criterion: str='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.feature_names = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        self.root = self.tree_grow(X, y)

    def tree_grow(self, X, y, current_depth: int=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        node = Node(n_samples=n_samples)

        if (self.max_depth is not None and current_depth >= self.max_depth) or \
           (n_classes == 1) or \
           (n_samples < self.min_samples_split):
            node.is_leaf = True
            node.value = self.most_common_label(y)
            return node
        
        # Find the best split
        best_feature_index, best_threshold, gain = self.find_best_split(X, y)

        if gain <= 0:
            node.is_leaf = True
            node.value = self.most_common_label(y)
            return node
        
        node.is_leaf = False
        node.feature_index = best_feature_index
        node.threshold = best_threshold
        node.gain = gain

        left_idx = X[:, best_feature_index] <= best_threshold
        right_idx = ~left_idx

        node.left = self.tree_grow(X[left_idx], y[left_idx], current_depth+1)
        node.right = self.tree_grow(X[right_idx], y[right_idx], current_depth+1)
        
        return node

    def find_best_split(self, X, y):
        greatest_gain = -np.inf
        best_feature = None
        best_thresh = None

        n_features = X.shape[1]
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                gain = self.information_gain(y[left_mask], y[right_mask])
                
                if gain > greatest_gain:
                    greatest_gain = gain
                    best_feature = feature_index
                    best_thresh = threshold

        return best_feature, best_thresh, greatest_gain
    
    def information_gain(self, y_left, y_right): # left and right samples
        n_left = len(y_left) # number of left samples
        n_right = len(y_right) # number of right samples
        n_total = n_left + n_right # total samples count

        if n_total == 0:
            return 0

        # probabilities
        p_left = n_left / n_total 
        p_right = n_right / n_total

        # left and right entropy
        entropy_left = self.calculate_entropy(y_left)
        entropy_right = self.calculate_entropy(y_right)

        # entropy before and after split
        parent_entropy = self.calculate_entropy(np.concatenate([y_left, y_right]))
        child_entropy = p_left * entropy_left + p_right * entropy_right

        return parent_entropy - child_entropy # information gain
    
    def calculate_entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def most_common_label(self, y):
        if len(y) == 0:
            return 0
        return np.bincount(y).argmax()
        
    def predict_sample(self, xi, node):
        if node.is_leaf:
            return node.value
        
        if xi[node.feature_index] <= node.threshold:
            return self.predict_sample(xi, node.left)
        else:
            return self.predict_sample(xi, node.right)
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.predict_sample(x, self.root) for x in X])
    


    