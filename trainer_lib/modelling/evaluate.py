import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, confusion_matrix

class Evaluation:
    """
    Evaluate a fitted classifier.
    """

    def __init__(self, clf, X, y):
        self.clf = clf
        self.X = X
        self.y = y

    def get_plot_roc(self):
        """
        Get plot of Receiver operator curve.
        """
        clf, X, y = self.clf, self.X, self.y
        pl, ax = plt.subplots(figsize=(11,5))
        p = plot_roc_curve(clf, X, y, ax=ax)
        clf_name = list(clf.named_steps.keys())[-1]
        plt.title(f"Receiver Operator Curve for {clf_name} Classifier")
        plt.show()
        
    def get_confusion_matrix(self):
        """
        Calculate the confusion matrix and return
        """
        clf, X, y = self.clf, self.X, self.y
        preds = clf.predict(X)
        return confusion_matrix(y, preds)

    def get_f1_score(self, confusion_matrix):
        """
        Calculate the f1 score.
        """
        TP = confusion_matrix[0,0]
        TN = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        return TP / (TP + 0.5 * (FP + FN))

    def get_accuracy(self, confusion_matrix):
        """
        Calculate the accuracy
         (correct predictions as a proportion of all predictions).
        """
        TP = confusion_matrix[0,0]
        TN = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        return (TP + TN) / ( TP + TN + FP + FN) 

    def run(self, plot=False):
        """
        Take a classifier and test data and evaluate the performance.

        Returns a dict with evaluation metrics.
        """
        confusion_ = self.get_confusion_matrix()
        if plot:
            self.get_plot_roc()
        return {
            "f1_score": self.get_f1_score(confusion_),
            "accuracy": self.get_accuracy(confusion_)
        }