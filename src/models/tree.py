import numpy as np
import pandas as pd

# Node
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
# Decision Tree
class DecisionTreeClassifier:
    def __init__(self, criterion: str='entropy', max_depth: int=4, min_samples_split: int=2, min_samples_leaf: int=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def count_entropy(self, n_samples):
        # n_samples is a vector with the number of samples of each class
        n_samples = np.array(n_samples)
        n_total = np.sum(n_samples) # total sum of samples
        if n_samples.dtype == np.iterable:
            probabilities = [ni / n_total for ni in n_samples]
        else:
            probabilities = [n_samples / n_total]
        H = -np.sum( pi * np.log2(pi) for pi in probabilities )
        return H
    
    def information_gain(self, n_left, n_right):
        # probabilities of left samples and right samples
        p_left = n_left / (n_left + n_right)
        p_right = n_right / (n_left + n_right)

        # right and left entropy
        entropy_left = self.count_entropy(n_left)
        entropy_right = self.count_entropy(n_right)

        # total entropy before split
        previous_entropy = entropy_left + entropy_right
        # total entropy after split
        conditional_entropy = p_left * entropy_left + p_right * entropy_right

        information_gain = previous_entropy - conditional_entropy
        return information_gain

    def find_best_split(self, X, y):
        # Find a feature and a threshold, the split by which gives the greatest information gain
        greatest_IG = -np.inf # the greatest gain
        best_feature = None # the best feature to split by it
        best_thresh = None # the best threshold to split by it

        n_features = X.shape[1]
        for feature_index in range(n_features):
            thresholds = X[:, feature_index]
            class_labels = y
            for thresh in thresholds:
                n_left = np.sum( thresholds <= thresh )
                n_right = np.sum( thresholds > thresh )
                new_IG = self.information_gain(n_left, n_right)
                if new_IG > greatest_IG:
                    greatest_IG = new_IG
                    best_feature = feature_index
                    best_thresh = thresh

        return best_feature, best_thresh
    
    def tree_grow(self, X, y, current_depth=0):
        n_samples = X.shape[0]
        
        # Check for conditions of stopping
        if n_samples < self.min_samples_split or len(np.unique(y)) == 1 or current_depth >= self.max_depth:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # Fing the best split
        best_feature_index, best_thresh = self.find_best_split(X, y)
        b = X[:, best_feature_index]
        left_indices = [int(i) for i in range(len(b)) if b[i] <= best_thresh]
        right_indices = [int(i) for i in range(len(b)) if b[i] > best_thresh]

        left_tree = self.tree_grow(X[left_indices], y[left_indices], current_depth+1)
        right_tree = self.tree_grow(X[right_indices], y[right_indices], current_depth+1)

        return Node(feature_index=best_feature_index, threshold=best_thresh, left=left_tree, right=right_tree)


    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.root = self.tree_grow(X, y)

    def predict(self, X):
        X = np.array(X)
        return [self._predict_sample(sample) for sample in X]

    def _predict_sample(self, sample):
        # Recurse predict class label for each sample
        node = self.root
        while node.value is None:
            if sample[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

