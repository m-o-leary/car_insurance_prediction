from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DatasetCleanerPipeline:
    """
    Class to manage cleaning if dataset where
    """

    MISSING_CONFIG = {
        "Communication": { "strategy": "constant", "fill_value": "other" },
        'Education': { "strategy": 'median' },
        'Outcome': { "strategy": "constant", "fill_value": int(0) }
    }

    def __init__(self, config=MISSING_CONFIG ):
        self.config = config

    def __repr__(self):
        return f"DatasetCleanerPipeline(config={str(self.config)})"
    
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, incoming_df, **transform_params):
        """
        Run the transform method of the pipeline on this class.
        """
        outgoing_df = incoming_df.copy()

        for feature,imputer_args in self.config.items():
            outgoing_df[feature] = SimpleImputer(**imputer_args).fit_transform(outgoing_df[[feature]])

        return outgoing_df