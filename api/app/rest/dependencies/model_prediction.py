import joblib

def load_pipeline(model_id, *args):
    """
    Load a model.
    
    Full pipeline training and having fit_transform / predict / predict_proba methods

    Models are expected to be located in the /models directory

    Parameters
    ----------
    model_id : Str
        Model hash id to load.
    """

    pipeline = joblib.load(f"/models/{model}.pkl")

    return pipeline

def map_call_line_to_prediction(call, *args):
    """
    Load a model.
    
    Full pipeline training and having fit_transform / predict / predict_proba methods

    Models are expected to be located in the /models directory

    Parameters
    ----------
    model_id : Str
        Model hash id to load.
    """

    pipeline = joblib.load(f"/models/{model}.pkl")

    return pipeline