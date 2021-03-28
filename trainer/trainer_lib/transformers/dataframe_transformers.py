from sklearn.pipeline import Pipeline
from collections import defaultdict
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class SelectColumnsTransfomer(TransformerMixin, BaseEstimator):
    """
    Class to select columns from a dataframe in a pipeline.
    """
    def __init__(self, columns=None):
        """
        Initialize transformer with the list of columns to keep.
        """
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        """
        Select only columns in self.columns and return.
        """
        cpy_df = X[self.columns].copy()
        return cpy_df

class CategoricalLimitTransformer(TransformerMixin, BaseEstimator):
    """
    Class to transform categorical columns to only keep the top N levels.
    """
    def __init__(self, n, columns=[]):
        self.n = n
        self.columns = columns
        self.maps = {}

    def get_top_n(self,X):
        """
        Get the top n levels in the columns in the DF

        :param X: Dataframe containing colums which need to be limited
        :type X: pandas.DataFrame
        """
        for col in self.columns:
            __levels = list(X[col].value_counts()[:self.n].index)
            __mapper_dict = defaultdict(lambda: "other")
            for level in __levels:
                __mapper_dict[level] = level
            self.maps[col] = __mapper_dict

    def transform(self, incoming_df, **transform_params):
        """
        Transform the columns specified on init to have n+1 levels.

        The top-n levels will remain the same but the +1 level will be an 'other' level.
        """
        for col in self.columns:
            incoming_df[col] = incoming_df[col].map( self.maps[col])

        return incoming_df

    def fit(self, X, y=None, **fit_params):
        """
        Find the top-n levels of the columns supplied at init.
        """
        self.get_top_n(X)
        return self

class CallDurationTransformer(TransformerMixin, BaseEstimator):
    """
    Transform CallStart and CallEnd times to a duration in mins.
    """

    def __init__(self, fmt='%H:%M:%S', start_column="CallStart", end_column="CallEnd"):
        self.fmt = fmt
        self.start = start_column
        self.end = end_column
    
    def fit(self, *args, **kwargs):
        return self

    def get_time_delta(self,row):
        """
        Calculate the time delta between 2 columns in minutes.
        """
        __delta = datetime.strptime(row[self.end], self.fmt) \
                - datetime.strptime(row[self.start], self.fmt)
        return __delta.total_seconds() / 60 

    def transform(self, incoming_df, **transform_params):
        """
        Transform the columns specified on init to have n+1 levels.

        The top-n levels will remain the same but the +1 level will be an 'other' level.
        """
        

        incoming_df["CallDurationMins"] = incoming_df.apply(self.get_time_delta, axis=1)

        return incoming_df

class TimeOfDayTransformer(TransformerMixin, BaseEstimator):
    """
    Add column to dataframe which is a categorical variable  the time of day.
    """

    def __init__(self, fmt='%H:%M:%S', column="CallStart"):
        self.fmt = fmt
        self.column = column

    def fit(self, *args, **kwargs):
        return self

    def get_time_of_day(self, time_stamp):
        """
        Parse a timestamp in the format of self.fmt and return the time of day as a string.
        """
        hour = datetime.strptime(time_stamp, self.fmt).hour
        
        if hour < 12:
            return 'morning'
        elif hour > 18:
            return 'evening'
        else:
            return 'afternoon'

    def transform(self, incoming_df, **transform_params):
        """
        Add a column to the incoming_df containing the time of day of the call.

        Time of day levels are:

            morning = time < 12:00
            afternoon = 12:00 < time < 18:00 
            evening = time > 18:00 
        """
        incoming_df['CallTimeOfDay'] = incoming_df[self.column].apply(self.get_time_of_day)
        return incoming_df

class MonthNameTransformer:
    """
    Class to transform a 3-letter month abbreviation into an ordinal integer column.
    """

    MONTH__MAP = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }

    def __init__(self, column="LastContactMonth", map_dict=MONTH__MAP):
        self.column = column
        self.map = defaultdict(lambda: -1 )
        self.map.update(map_dict)

    def fit(self, *args, **kwargs):
        return self
        
    def transform(self, incoming_df, **tranform_kwargs):
        """
        Transform the self.column which should be a column of
         strings corresponding to the keys available in self.MONTH__MAP to a numeric variable.
        """
        incoming_df[self.column] = incoming_df[self.column].map(self.map)
        return incoming_df