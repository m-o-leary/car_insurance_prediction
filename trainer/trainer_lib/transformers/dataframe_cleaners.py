from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class DatasetCleanerTransformer:
    """
    Class to manage cleaning of kaggle dataset.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('communication', FunctionTransformer(
                func=stem_str, 
                kw_args={'stemmer': RSLPStemmer()})),
            ('education', FunctionTransformer(
                func=stem_str, 
                kw_args={'stemmer': RSLPStemmer()})),
            ('outcome', FunctionTransformer(
                func=stem_str, 
                kw_args={'stemmer': RSLPStemmer()}))
        ])
    
    def clean_communication(self, input_df):
        """
        Clean the communication column in the data.

        This will fill NA values with the string "missing"
        """

        input_df['Communication'].fillna("missing", inplace=True)
    
    def clean_education(self, input_df):
        """
        Clean the educatkln column in the data.

        This will fill NA values with the string "missing"
        """

        input_df['Communication'].fillna("missing", inplace=True)

    def fit(*args, **kwargs):
        return self

    def transform(self, incoming_df, **transform_params):
        """
        Fill the 
        """