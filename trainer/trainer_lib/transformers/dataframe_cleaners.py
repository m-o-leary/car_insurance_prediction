from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class DatasetCleanerTransformer:
    """
    Class to manage cleaning of kaggle dataset.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('communication', FunctionTransformer(
                func=self.clean_communication )),
            ('education', FunctionTransformer(
                func=self.clean_education )),
            ('outcome', FunctionTransformer(
                func=self.clean_outcome ))
        ])
    
    def clean_communication(self, input_df):
        """
        Clean the communication column in the data.

        This will fill NA values with the string "missing"
        """

        input_df['Communication'].fillna("missing", inplace=True)
        return input_df
    
    def clean_education(self, input_df):
        """
        Clean the educatkln column in the data.

        This will fill NA values with the string "missing"
        """

        input_df['Education'].fillna("unknown", inplace=True)
        return input_df

    def clean_outcome(self, input_df):
        """
        Clean the outcome column in the data.

        This will fill NA values with the string "not_contacted"
        """

        input_df['Outcome'].fillna("not_contacted", inplace=True)
        return input_df

    def fit(self, X, y=None, **kwargs):
        """
        Fit the pipeline on the class.
        """
        self.pipeline.fit(X)
        return self

    def transform(self, incoming_df, **transform_params):
        """
        Run the transform method of the pipeline on this class.
        """
        return self.pipeline.transform(incoming_df)