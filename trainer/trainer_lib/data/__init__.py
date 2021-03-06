from .config import DATA_URL, TRAIN_DATA, TEST_DATA, TARGET_VARIABLE
from .data_loader import KaggleCarInsuranceDataLoader
from .data_profiler import DataProfiler
from sklearn.model_selection import train_test_split

class DataManager:
    """
    Manager class to load and profile data.
    """

    def __init__(self, save_path, report_path, refresh=False, split=0.75, seed=43):
        """
        Initialize object and download data.

        :param save_path: Location where data will be saved.
        :type save_path: str
        :param report_path: Location where reports will be saved.
        :type report_path: str
        """
        self.loader = KaggleCarInsuranceDataLoader(save_path, fetch=refresh)
        self.report_path = report_path
        self.split = split
        self.seed = seed
        self.train_x, self.train_y = self.loader.get_train_data(return_x_y=True)
        self.test_x, self.test_y = self.loader.get_test_data(return_x_y=True)
        
    @property
    def train(self):
        return self.train_x, self.train_y
    
    @property
    def test(self):
        return self.test_x, self.test_y

    @property
    def train_test(self):
        """
        Return the training and test sets split from the training csv.

        :return: tuple(X_train, X_test, y_train, y_test)
        :rtype: tuple
        """
        X, y = self.train
        return train_test_split(
            X,y, 
            random_state=self.seed, 
            train_size=self.split)

    def profile(self):
        """
        Profile both train and test and save reports.
        """
        train = self.loader.get_train_data(return_x_y=False)
        test = self.loader.get_test_data(return_x_y=False)

        self.train_profiler = DataProfiler(data=train, 
                                           report_title='train_data', 
                                           out_path=self.report_path)

        self.test_profiler = DataProfiler(data=test, 
                                          report_title='test_data', 
                                          out_path=self.report_path)
        self.train_profiler.profile()
        self.test_profiler.profile()