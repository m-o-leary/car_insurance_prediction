import requests
import pandas as pd
import json
from trainer_lib.modelling.model_config import KEYS_MAP, PREDICTION_URL
from rich.console import Console

console = Console()

class Predict:
    """
    Class to get a prediction from the REST API serving out model.
    """
    def __init__(self, rows, model_id, endpoint=PREDICTION_URL):
        self.rows = rows
        self.model_id = model_id
        self.endpoint = endpoint


    def __call__(self):
        """
        Call this class to get a prediction.
        """

        # Saving to DB
        headers = {
            'Content-Type': 'application/json'
        }

        # Handle when the data passed is either a single row or a DF
        data = [ DataMapper(row.dropna().to_dict()).map() for i, row in self.rows.iterrows() ] \
            if type(self.rows) == pd.core.frame.DataFrame \
                else [ DataMapper(self.rows.dropna().to_dict()).map() ]
                
        # Add to our DB - the FastAPI container must be running!
        x = requests.post( 
            f"{self.endpoint}{self.model_id}", 
            headers = headers, 
            data = json.dumps(data) )
        
        # console.print(x.text)
        return x
        
class DataMapper:
    """
    Class to map from raw data to API schema.

    Schema is available here: http://localhost:3007/redoc#operation/run_prediction_insurance_model_predictions__model_id__post
    """

    def __init__(self, old_dict, key_map=KEYS_MAP):
        self.key_map = key_map
        self.old_dict = old_dict

    def map(self):
        return { 
            self.key_map[old_key] : str(val)
            for old_key, val in self.old_dict.items() }