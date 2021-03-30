import hashlib
import pandas as pd
import joblib

def get_hash_from_data(df):
    """
    Generates and returns a hash of the provided dataframe.
    """
    __hash = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest() 
    return __hash