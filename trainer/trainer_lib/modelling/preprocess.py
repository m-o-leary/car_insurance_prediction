# Load data
# Fill missing
# Feature engineering
# Store processed data
from sklearn.pipeline import Pipeline, FeatureUnion
    from trainer_lib.transformers import SelectFeaturesTransfomer
from trainer_lib.transformers import CallDurationTransformer
from trainer_lib.transformers import TimeOfDayTransformer
from trainer_lib.transformers import MonthNameTransformer
from trainer_lib.transformers import EducationTransformer
from trainer_lib.transformers import DatasetCleanerPipeline
from trainer_lib.transformers import OutcomeTransformer
from trainer_lib.transformers import JobTransformer
from trainer_lib.data.config import ALL_COLUMNS




# Get categorical and numeric column names
# Feature engineering pipeline

feature_engineering = Pipeline([
    ("add_time_duration", CallDurationTransformer()),
    ("add_time_of_day", TimeOfDayTransformer()),
    ("convert_job", JobTransformer(),
    ("convert_month", MonthNameTransformer()),
    ("convert_education", EducationTransformer()),
    ("convert_outcome", OutcomeTransformer()),
    ("impute_missing", DatasetCleanerPipeline()),
    ("column_selection", SelectFeaturesTransfomer(features=ALL_COLUMNS))
])

