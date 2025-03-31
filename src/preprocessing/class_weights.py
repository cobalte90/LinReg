import numpy as np
def compute_class_weight(y):
    classes, n_samples = np.unique(y, return_counts=True)
    n_classes = len(classes)
    w = {}
    for i in range(n_classes):
        w[classes[i]] = len(y) / (n_classes * n_samples[i])
    return w