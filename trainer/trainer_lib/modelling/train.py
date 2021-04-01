# Transformers
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from rich.console import Console
from rich.live import Live

# Utils
from trainer_lib.utils.output import TrainerConsole
from trainer_lib.data.saving import ModelSaver
from trainer_lib.data.data_reconstructor import DataReconstructor

# Setup
np.random.seed(12345)
console = Console()

# Our code
from trainer_lib import DataManager
from trainer_lib.transformers import SelectFeaturesTransfomer
from trainer_lib.transformers import CallDurationTransformer
from trainer_lib.transformers import TimeOfDayTransformer
from trainer_lib.transformers import MonthNameTransformer
from trainer_lib.transformers import EducationTransformer
from trainer_lib.transformers import DatasetCleanerPipeline
from trainer_lib.transformers import OutcomeTransformer
from trainer_lib.transformers import JobTransformer
from trainer_lib.transformers import DaysPassedTransformer
from trainer_lib.modelling.evaluate import Evaluation
from trainer_lib.modelling.explain import Explain
from trainer_lib.data.config import ALL_COLUMNS, ONE_HOT_CATEGORICAL_COLUMNS, SCALABLE_NUMERIC_COLUMNS

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV

class PreProcessor:
    
    PRE_PROCESSING_STEPS = [
        ("add_time_duration", CallDurationTransformer()),
        ("add_time_of_day", TimeOfDayTransformer()),
        ("convert_job", JobTransformer()),
        ("convert_month", MonthNameTransformer()),
        ("convert_education", EducationTransformer()),
        ("convert_outcome", OutcomeTransformer()),
        ('replace_negative_days_passed', DaysPassedTransformer()),
        ("impute_missing", DatasetCleanerPipeline()),
        ("column_selection", SelectFeaturesTransfomer(features=ALL_COLUMNS))
    ]
    
    def __init__(self,steps=PRE_PROCESSING_STEPS):
        self.steps = steps

    def make_pipeline(self):
        self.pipeline = Pipeline(self.steps)
        return self.pipeline

class Encoder:

    def __init__(self,
                 one_hot_features=ONE_HOT_CATEGORICAL_COLUMNS, 
                 scale_features=SCALABLE_NUMERIC_COLUMNS,
                 categorical_encoder=OneHotEncoder,
                 numeric_encoder=StandardScaler):

        self.one_hot_features = one_hot_features
        self.scale_features = scale_features

        self.categorical_preprocessing = Pipeline([('cat', categorical_encoder())])
        self.numerical_preprocessing = Pipeline([('numeric', numeric_encoder())])

    def make_pipeline(self):

        self.pipeline = ColumnTransformer([
            ('categorical', self.categorical_preprocessing, self.one_hot_features),
            ('numeric', self.numerical_preprocessing, self.scale_features)
        ], remainder='passthrough')

        return self.pipeline

class Trainer:
    """
    Trainer class to train multiple models and evaluate.
    """

    def __init__(self, X, y, grid, save_path):
        """
        Instantiate the trainer onject with all required data.

        :param X: Features dataframe to fit classifier on.
        :type X: pandas.DataFrame
        :param y: Target Variable
        :type y: pandas.Series / numpy.ndarray  
        :param grid: A dict of classifier names as keys with
         a tuple of (classifier, grid search dict) as corresponding values,
        :type grid: dict
        :param save_path: Location to save pickled model,
        :type save_path: str
        """
        self.X = X
        self.y = y
        self.grid = grid
        self.data_pipeline = Pipeline([
            ( 'feature_engineering', PreProcessor().make_pipeline()),
            ( "encoder", Encoder().make_pipeline()),
        ])
        self.save_path = save_path

    def train(self, experiment_name):
        """
        Train all the classifiers passed as args to this class (or ttbe default list).

        :param experiment_name: Name for the experiment in the database.
        :type experiment_name: str
        :return: dict containing the classifiers and their score.
        :rtype: [type]
        """
        self.best_score = 0
        self.best_classifier = None
        self.performance={}
        
        console.print()
        console.rule(f"Experiment name: {experiment_name}")
        console.print()
        
        with Live(TrainerConsole(self).update(), refresh_per_second=4) as live:
            for name, __ in self.grid.items():
                
                model, grid = __

                # Create model pipeline with xgboost classifier
                clf = Pipeline([
                    ("pre_processing", self.data_pipeline),
                    ('model', model)
                ])

                best_estimator, score, params = self.grid_search(clf, grid, self.X, self.y)
                # console.print(best)
                if score > self.best_score:
                    self.best_score = score
                    self.best_classifier = best_estimator
                    self.best_classifier__name = name
                    self.best_paramas = params

                hash_id = ModelSaver(self.X, self.y, self.save_path).save_best(f"{experiment_name}__{name}", best_estimator, params)
                self.performance[name] = (score, params, hash_id)
                live.update(TrainerConsole(self).update(name))

    def grid_search(self, clf, param_grid, X, y, k=10, verbose=0, scoring='roc_auc'):
        """
        Perform grid search on classifier arcoss paramater grid

        :param clf: sklearn compatible classifier
        :type clf: sklearn compatible classifier
        :param param_grid: Parameter grid to run over.
        :type param_grid: dic
        :param X: Features
        :type X: pandas.Dataframe
        :param y: target variable
        :type y: pandas.Series / numpy.ndarray
        :return: Tuple containing: best estimator, average score over k-folds, best parameters
        :rtype: Tuple
        """
        grid = GridSearchCV(
            clf, 
            param_grid, 
            cv=k, 
            n_jobs=-1, 
            verbose=verbose,
            scoring=scoring)

        # Fit 
        grid.fit(X, y)   

        mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]
        std_score = grid.cv_results_["std_test_score"][grid.best_index_]

        grid.best_params_, mean_score, std_score

        return (grid.best_estimator_, mean_score, grid.best_params_)

   