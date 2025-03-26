class ClassificationMetrics:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def confusion(self, y_true, y_pred):
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                if y_pred[i] == 1:
                    self.true_positive += 1
                if y_pred[i] == 0:
                    self.true_negative += 1
            else:
                if y_pred[i] == 1:
                    self.false_positive += 1
                if y_pred[i] == 0:
                    self.false_negative += 1
        return f"TP = {self.true_positive}\nTN = {self.true_negative}\nFP = {self.false_positive}\nFN = {self.false_negative}"

    def my_accuracy_score(self, y_true, y_pred):
        self.confusion(y_true, y_pred)
        score = (self.true_positive + self.true_negative) / (self.true_positive + self.false_negative + self.true_negative + self.false_positive)
        return score

    def my_precision_score(self, y_true, y_pred):
        self.confusion(y_true, y_pred)
        score = self.true_positive / (self.true_positive + self.false_positive)
        return score

    def my_recall_score(self, y_true, y_pred):
        self.confusion(y_true, y_pred)
        score = self.true_positive / (self.true_positive + self.false_negative)
        return score

    def my_f1_score(self, y_true, y_pred, beta=1):
        precision = self.my_precision_score(y_true, y_pred)
        recall = self.my_recall_score(y_true, y_pred)
        score = (1 + beta**2) * (precision * recall / (beta**2 * precision + recall))
        return score

