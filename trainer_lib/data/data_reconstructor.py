import pandas as pd
from trainer_lib.data.config import ONE_HOT_CATEGORICAL_COLUMNS, ALL_COLUMNS

class DataReconstructor:
    """
    Class to reconstruct a processed dataset into a pandas dataframe.
    """
    def __init__(self, X, clf, cat_features=ONE_HOT_CATEGORICAL_COLUMNS):
        self.X = X
        self.clf = clf
        self.cat_features = cat_features

    def make(self):
        names = list(
            self.clf['pre_processing']['encoder']
            .transformers_[0][1]['cat']
            .get_feature_names(self.cat_features))
            
        names.extend([c for c in ALL_COLUMNS if c not in self.cat_features])

        return pd.DataFrame(self.clf.named_steps['pre_processing'].transform(self.X), columns=names)
