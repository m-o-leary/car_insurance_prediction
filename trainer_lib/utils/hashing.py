import hashlib
import pandas as pd

class Hasher:
    """
    Class to handle hashing for the project.
    """
    @staticmethod
    def get_hash_from_data(df):
        """
        Generates and returns a hash of the provided dataframe.
        """
        __hash = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
        return __hash

    @staticmethod
    def get_hash_from_pipeline(pipeline):
        """
        Generates and returns a hash of the provided pipeline.
        """
        __hash = hashlib.sha1(str(pipeline).encode()).hexdigest()
        return __hash

    @staticmethod
    def get_hash_from_object(obj):
        """
        Hash any python object which implements the __str__ or __repr__ function.
        :param obj: The object to return a hash for.
        :return: str
        """
        __hash = hashlib.sha1(str(obj).encode()).hexdigest()
        return __hash
