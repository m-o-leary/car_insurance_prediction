# Transformers
import numpy as np
from rich.console import Console
from sklearn.pipeline import Pipeline, FeatureUnion
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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from trainer_lib.data.config import ALL_COLUMNS, ONE_HOT_CATEGORICAL_COLUMNS, SCALABLE_NUMERIC_COLUMNS

# Classifiers
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, GridSearchCV

# Evaluation
from sklearn.metrics import plot_roc_curve, confusion_matrix
from sklearn.model_selection import KFold
import shap
import pandas as pd
import matplotlib.pyplot as plt

class PreProcessor:
    
    PRE_PROCESSING_STEPS = [
        ("add_time_duration", CallDurationTransformer()),
        ("add_time_of_day", TimeOfDayTransformer()),
        ("convert_job", JobTransformer()),
        ("convert_month", MonthNameTransformer()),
        ("convert_education", EducationTransformer()),
        ("convert_outcome", OutcomeTransformer()),
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
    Trainer class
    """

    NAMES = [
        # "Linear SVM",
        # "Decision Tree", 
        # "Random Forest", 
        # "Neural Net", 
        # "AdaBoost",
        # "Naive Bayes",
        "XGB"
    ]

    CLASSIFIERS = [
        # SVC(kernel="linear", C=0.025),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        XGBClassifier(objective='binary:logistic', eval_metric='error')
    ]

    GRID_SEARCH = {
        "Linear SVM": {},
        "Decision Tree": {}, 
        "Random Forest": {}, 
        "Neural Net": {}, 
        "AdaBoost": {},
        "Naive Bayes": {},
        "XGB": {
            'model__max_depth': [2, 3, 5, 7, 10],
            'model__n_estimators': [10, 100, 500],
        }
    }

    def __init__(self, names=NAMES, clfs=CLASSIFIERS, grid=GRID_SEARCH):
        """
        Instantiate the Trainer classs

        :param names: Classifier names, defaults to NAMES
        :type names: list of strings, optional
        :param clfs: Classifiers to fit to the data, defaults to CLASSIFIERS
        :type clfs: list of sklearn estimator instances, optional
        """

        self.names = names
        self.clfs = clfs
        self.grid = grid
        self.data_pipeline = Pipeline([
            ( 'feature_engineering', PreProcessor().make_pipeline()),
            ( "encoder", Encoder().make_pipeline()),
        ])

    def train(self, X, y):
        """
        Train all the classifiers associated with the 

        :param X: Feature dataframe.
        :type X: pandas.DataFrame
        :param y: Target series
        :type y: pandas.Series or np.ndarray
        :return: List of tuples containing the name of each classifier and their score.
        :rtype: List
        """
        self.best_score = 0
        self.best_classifier = None
        self.performance={}

        kf = KFold(n_splits=10)
        kf.get_n_splits(X)

        for name, model in zip(self.names, self.clfs):
            

            # Create model pipeline with xgboost classifier
            console.rule(name)

            clf = Pipeline([
                ("pre_processing", self.data_pipeline),
                ('model', model)
            ])

            scores = []
            
            score = self.grid_search(clf, self.GRID_SEARCH[name], X, y)

            if score > self.best_score:
                self.best_score = score
                self.best_classifier = clf
                self.best_classifier__name = name
            self.performance[name] = score
        
        return self.performance

    def grid_search(self,clf, param_grid, X, y):
        console.print("Starting grid search...")
        
        grid = GridSearchCV(
            clf, 
            param_grid, 
            cv=10, 
            n_jobs=-1, 
            verbose=5,
            scoring='roc_auc')

        grid.fit(X, y)   

        mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]
        std_score = grid.cv_results_["std_test_score"][grid.best_index_]

        grid.best_params_, mean_score, std_score

        console.log(f"Best parameters: {grid.best_params_}")
        console.log(f"Mean CV score: [green]{mean_score: .6f}[/green]")
        console.log(f"Standard deviation of CV score: {std_score: .6f}")
        return mean_score


        
class Evaluation:
    """
    Evaluate a fitted classifier.
    """
    def get_plot_roc(self, clf, X, y):
        """
        Get plot of Receiver operator curve.
        """
        pl, ax = plt.subplots(figsize=(11,5))
        p = plot_roc_curve(clf, X, y, ax=ax)
        clf_name = list(clf.named_steps.keys())[-1]
        plt.title(f"Receiver Operator Curve for {clf_name} Classifier")
        plt.show()
        
    def get_confusion_matrix(self, clf, X, y):
        """
        Calculate the confusion matrix and return
        """
        preds = clf.predict(X)
        return confusion_matrix(y, preds)

    def get_f1_score(self, confusion_matrix):
        """
        Calculate the f1 score.
        """
        TP = confusion_matrix[0,0]
        TN = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        return TP / (TP + 0.5 * (FP + FN))

    def get_accuracy(self, confusion_matrix):
        """
        Calculate the accuracy
         (correct predictions as a proportion of all predictions).
        """
        TP = confusion_matrix[0,0]
        TN = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        FN = confusion_matrix[1,0]
        return (TP + TN) / ( TP + TN + FP + FN) 

    def evaluate(self, *classifier_and_data):
        """
        Take a classifier and test data and evaluate the performance.

        Returns a dict with evaluation metrics.
        """
        confusion_ = self.get_confusion_matrix(*classifier_and_data)

        return self.get_plot_roc(*classifier_and_data), {
            "f1_score": self.get_f1_score(confusion_),
            "accuracy": self.get_accuracy(confusion_)
        }

class Explain:


    def __init__(self, clf, X ):

        self.clf = clf
        self.X = X
        self.explainer = shap.Explainer(clf.steps[-1][1])
        self.make_dataframe()
        self.shap_values = self.explainer(self.X_df)

    def make_dataframe(self):
        names = list(
            self.clf['pre_processing']['encoder']
            .transformers_[0][1]['cat']
            .get_feature_names(ONE_HOT_CATEGORICAL_COLUMNS))
            
        names.extend([c for c in ALL_COLUMNS if c not in ONE_HOT_CATEGORICAL_COLUMNS])
        self.X_df = pd.DataFrame(self.clf.named_steps['pre_processing'].transform(self.X), columns=names)

    def waterfall(self, index=0):
        shap.plots.waterfall(self.shap_values[index], max_display=15)

    def prediction(self, index=0):
        p = shap.plots.force(self.shap_values[index])
        p.matplotlib((30,8),True, 35)