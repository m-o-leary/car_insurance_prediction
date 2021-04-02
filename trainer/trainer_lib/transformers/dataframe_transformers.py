from collections import defaultdict
from datetime import datetime
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin

class SelectFeaturesTransfomer(BaseEstimator, TransformerMixin):
    """
    Class to select features from a dataframe in a pipeline.
    """
    def __init__(self, features=None):
        """
        Initialize transformer with the list of features to keep.
        """
        self.features = features

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        """
        Select only columns in self.columns and return.
        """
        outgoing_df = X[self.features].copy()
        return outgoing_df

class CategoricalLimitTransformer(BaseEstimator, TransformerMixin):
    """
    Class to transform categorical features to only keep the top N levels.
    """
    def __init__(self, n, features=[]):
        self.n = n
        self.features = features
        self.maps = {}

    def get_top_n(self,X):
        """
        Get the top n levels in the columns in the DF

        :param X: Dataframe containing colums which need to be limited
        :type X: pandas.DataFrame
        """
        for col in self.features:
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
        outgoing_df = incoming_df.copy()
        for col in self.features:
            outgoing_df[col] = outgoing_df[col].map( self.maps[col] )

        return outgoing_df

    def fit(self, X, y=None, **fit_params):
        """
        Find the top-n levels of the columns supplied at init.
        """
        self.get_top_n(X)
        return self

class CallDurationTransformer(BaseEstimator, TransformerMixin):
    """
    Transform CallStart and CallEnd times to a duration in mins.
    """

    def __init__(self, fmt='%H:%M:%S', start_column="CallStart", end_column="CallEnd", new_feature_name="CallDurationMins"):
        self.fmt = fmt
        self.start_column = start_column
        self.end_column = end_column
        self.new_feature_name = new_feature_name
    
    def fit(self, *args, **kwargs):
        return self

    def get_time_delta(self,row):
        """
        Calculate the time delta between 2 columns in minutes.
        """
        __delta = datetime.strptime(row[self.end_column], self.fmt) \
                - datetime.strptime(row[self.start_column], self.fmt)
        return __delta.total_seconds() / 60 

    def transform(self, incoming_df, **transform_params):
        """
        Transform the columns specified on init to have n+1 levels.

        The top-n levels will remain the same but the +1 level will be an 'other' level.
        """

        outgoing_df = incoming_df.copy()
        outgoing_df["CallDurationMins"] = outgoing_df.apply(self.get_time_delta, axis=1).astype(int)

        return outgoing_df

class TimeOfDayTransformer(BaseEstimator, TransformerMixin):
    """
    Add column to dataframe which is a categorical variable  the time of day.
    """

    def __init__(self, fmt='%H:%M:%S', feature="CallStart", new_feature_name="CallTimeOfDay"):
        self.fmt = fmt
        self.feature = feature
        self.new_feature_name = new_feature_name

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
        outgoing_df = incoming_df.copy()
        outgoing_df[self.new_feature_name] = outgoing_df[self.feature].apply(self.get_time_of_day)
        return outgoing_df

class BaseMappingTransformer(BaseEstimator, TransformerMixin):
    """
    Base transformer class for mapping a series to other values.
    """

    def __init__(self, feature=None, map_dict=None, default_value=None, coerce='int'):
        self.feature = feature
        self.default_value = default_value
        self.map = map_dict
        self.coerce = coerce

    def __repr__(self):
        return f"BaseMappingTransformer(map={json.dumps(self.map).encode()})"

    def fit(self, *args, **kwargs):
        return self

    def transform(self, incoming_df, **tranform_kwargs):
        """
        Transform the self.column which should be a column of
         strings corresponding to the keys available in self.map to a numeric variable.
        """
        outgoing_df = incoming_df.copy()
        outgoing_df[self.feature] = outgoing_df[self.feature]\
            .map(self.map)\
            .fillna(self.default_value) \
            .astype(self.coerce, errors='ignore')

        return outgoing_df

class MonthNameTransformer(BaseMappingTransformer):
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

    def __init__(self, feature="LastContactMonth", map_dict=MONTH__MAP):
        self.map_dict =  map_dict
        super().__init__(feature=feature, map_dict=self.map_dict, default_value=np.nan)

class EducationTransformer(BaseMappingTransformer):
    """
    Class to transform a the education column into ordinal column.
    """

    EDU__MAP = {
        'primary': 1,
        'secondary': 2,
        'tertiary': 3
    }

    def __init__(self, feature="Education", map_dict=EDU__MAP):
        self.map_dict = map_dict
        super().__init__(feature=feature, map_dict=self.map_dict, default_value=np.nan)

class DaysPassedTransformer(BaseEstimator, TransformerMixin):
    """
    Class to transform a the DaysPassed feature so that the -1 values will be set to the median.
    """
    
    def __init__(self, feature="DaysPassed", strategy='median'):
        self.feature = feature
        self.strategy = strategy

    def fit(self, X, y=None, **kwargs):
        exec(f"self.fill_value = X.loc[X[self.feature] >= 0, self.feature].{self.strategy}()")
        return self

    def transform(self, incoming_df, y=None, **kwargs):
        """
        Transform the dataframe by filling all negative values with the median
        """
        outgoing_df = incoming_df.copy()
        mask = outgoing_df[self.feature] < 0
        outgoing_df.loc[mask, self.feature ] = self.fill_value
        return outgoing_df
        
class OutcomeTransformer(BaseMappingTransformer):
    """
    Class to transform a the Outcome column into binary column of 1 = success, 0 = otherwise.

    Decision is being made to have the outcome column be an indication of successful previous outcome only.

    This means that missing values are treated as "not a previous successful campaign regardless of whether or not
     the customer was previously contacted.
    """

    OUTCOME__MAP = {
        "failure": 0,
        "other": 0,
        "success": 1
    }

    def __init__(self, feature="Outcome", map_dict=OUTCOME__MAP):
        self.map_dict = map_dict
        super().__init__(feature=feature, map_dict=self.map_dict, default_value=np.nan)

class JobTransformer(BaseMappingTransformer):
    """
    Class to transform the Job feature into 3 levels instead of the high cardinality feature.
    """
    JOB_BUCKET__MAP = {
        'management': "professional",
        'self-employed': "professional",
        'entrepreneur': "professional",
        'blue-collar': "skilled",
        'technician': "skilled",
        'services': "skilled",
        'admin.': "workforce",
        'retired': "workforce",
        'housemaid': "workforce",
        'unemployed': "other",
        'student': "other"
    }


    def __init__(self, feature="Job", map_dict=JOB_BUCKET__MAP):
        self.map_dict = map_dict
        super().__init__(feature=feature, map_dict=self.map_dict, default_value="other")
