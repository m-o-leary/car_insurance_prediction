import requests 
import joblib
import json
from datetime import date
from trainer_lib.data.data_reconstructor import DataReconstructor
from trainer_lib.modelling.evaluate import Evaluation
from trainer_lib.utils.hashing import Hasher
from trainer_lib.modelling.model_config import MODEL_POST_URL
from trainer_lib.utils.filesystem import is_dir, make_dir

class ModelSaver:
    """
    Class to manage saving model.
    """

    def __init__(self, X, y, save_path, api_url=MODEL_POST_URL):
        self.X = X
        self.y = y
        self.api_url = api_url
        self.save_path = save_path
        self.X_hash = Hasher.get_hash_from_data(self.X)

        # Check and create dir for data
        if is_dir(self.save_path) == False:
            make_dir(self.save_path)
 
    def save_best(self, name, clf, params):
        """
        Save a copy of the pipeline.

        Get a hash for the raw data, the hashed data and pipeline.

        :param name: Name for model tracking
        :type name: str
        :return: dict of saved responses.
        :rtype: dict
        """
        metrics = Evaluation(clf, self.X, self.y).run()
        data_frame_constructor = DataReconstructor(self.X, clf)
        
        # Hashes for the important parts of the pipeline
        hash_dict = {
            "pipeline_hash": Hasher.get_hash_from_pipeline(clf),
            "raw_hash": self.X_hash,
            "processed_hash": Hasher.get_hash_from_data(data_frame_constructor.make()),
        }

        hash_dict.update({'estimator_id': Hasher.get_hash_from_object(hash_dict)})
        model_id = f"{self.save_path}{hash_dict['estimator_id']}.pkl"
        
        # Save the pipeline to storage
        PipeLineSaver.persist_pipeline(clf, model_id)

        # Saving to DB
        save_object_ = {
            "estimator_id": hash_dict['estimator_id'],
            "name": name,
            "params": params,
            "f1_score": metrics['f1_score'],
            "accuracy": metrics['accuracy'],
            "d": str(date.today()),
            "raw_hash": hash_dict['raw_hash'],
            "processed_hash": hash_dict['processed_hash'],
            "pipeline_hash": hash_dict['pipeline_hash']
        }


        headers = {
            'Content-Type': 'application/json'
        }

        # Add to our DB - the FastAPI container must be running!
        x = requests.post( self.api_url, headers = headers, data = json.dumps(save_object_) )

        # if ~x.ok:
        #     console.log(f"Could not save model details to db: {x.text}")
        return hash_dict['estimator_id']

class PipeLineSaver:
    """
    Load and save sklearn compatible pipelines
    """

    @staticmethod
    def persist_pipeline(pipeline, save_path):
        """
        Persist a pipeline.
        """
        joblib.dump(pipeline, save_path)

    @staticmethod
    def load_pipeline(pipeline_path):
        """
        Load a saved pipeline.
        """
        joblib.load(pipeline_path)