import shap
import matplotlib.pyplot as plt

class Explain:
    """
    Model explainability class.

    This will only work for tree based models.
    """

    def __init__(self, clf, X, X_df=None ):

        self.clf = clf
        self.X = X
        self.explainer = shap.Explainer(clf.steps[-1][1])
        if X_df is None:
            self.X_df = DataReconstructor(self.X, self.clf).make()
        else:
            self.X_df = X_df
        self.shap_values = self.explainer(self.X_df)

    def waterfall(self, index=0):
        shap.plots.waterfall(self.shap_values[index], max_display=15)

    def prediction(self, index=0):
        p = shap.plots.force(self.shap_values[index])
        return p

    def summary(self, index=0):
        f = plt.figure()
        shap.summary_plot(self.shap_values, self.X_df)
        f.savefig(f"./{hasher.get_hash_from_pipeline(self.clf)}_summary_plot.png", bbox_inches='tight', dpi=600)
