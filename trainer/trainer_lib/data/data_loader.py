import requests
import zipfile
import os
import io
import pandas as pd
from rich.console import Console
from abc import ABC, abstractmethod

from .config import TRAIN_DATA, TEST_DATA, DATA_URL, TARGET_VARIABLE
from trainer_lib.utils.filesystem import is_dir, make_dir
# Initialize console for degug printing
console = Console()

class AbstractDataLoader(ABC):
    """
    Abstract class for data loading.
    """
    @abstractmethod
    def get_train_data(self, return_x_y=True):
        pass

    @abstractmethod
    def get_test_data(self, return_x_y=True):
        pass


class KaggleCarInsuranceDataLoader(AbstractDataLoader):
    """
    Data loader class to retrieve data from kaggle competition page.
    """

    def __init__(self, save_path, url=DATA_URL, target=TARGET_VARIABLE, fetch=True):
        """
        Initialize class and optionally retrieve data from remote url

        :param save_path: Location where data will be extracted to,
        :type save_path: str
        :param url: URL for downloading zip file, defaults to DATA_URL
        :type url: str, optional
        :param target: target variable name in the data, defaults to TARGET_VARIABLE
        :type target: str, optional
        :param fetch: Fetch the data on init, defaults to True
        :type fetch: bool, optional
        """
        self.target = target
        self.url = url
        self.save_path = save_path

        # Check and create dir for data
        if is_dir(self.save_path) == False:
            make_dir(self.save_path)
            self.download_data()

        if fetch:
            self.download_data()

    def download_data(self):
        """
        Download the data from kaggle competition to local folder.
        """
        console.log("Downloading data.")
        r = requests.get(self.url)

        if r.ok:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.save_path)
            console.log("Done.")
        else:
            console.print('Could not download file from remote url.', style="bold red")
    
    def split_data(self, data):
        """
        Split the data into an X (dataframe) and Y (series)
        """
        X,y = data.drop(self.target, axis=1), data[[self.target]]
        return X,y

    def get_train_data(self, return_x_y=True):
        """
        Get the training dataset

        :param return_x_y: This flag will determine whether or not the data is return as an X,y tuple or a single Dataframe, defaults to True
        :type return_x_y: bool, optional
        :return: Returns the data in either a single dataframe or a tuple
        :rtype: Dataframe or tuple(Dataframe, Series)
        """
        __train = pd.read_csv(os.path.join(self.save_path, TRAIN_DATA))
        if return_x_y:
            return self.split_data(__train)
        else:
            return __train

    def get_test_data(self,return_x_y=True):
        """
        Get the test dataset

        :param return_x_y: This flag will determine whether or not the data is return as an X,y tuple or a single Dataframe, defaults to True
        :type return_x_y: bool, optional
        :return: Returns the data in either a single dataframe or a tuple
        :rtype: Dataframe or tuple(Dataframe, Series)
        """
        __test = pd.read_csv(os.path.join( self.save_path, TEST_DATA ))
        if return_x_y:
            return self.split_data(__test)
        else:
            return __test

    def get_train_test(self, *args, **kwargs):
        """
        Wrapper method to combine the train and test data retrieval.

        :return: a tuple of 
        :rtype: [type]
        """
        return self.get_train_data(*args, **kwargs), self.get_test_data(*args, **kwargs)