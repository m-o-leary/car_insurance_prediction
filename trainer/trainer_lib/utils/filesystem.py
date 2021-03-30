import os
import joblib

def make_dir(path):
    """
    Take a path and create it and any subdirectories

    Example:
        path = "/tmp/year/month/week/day"
        make_dir(path)

    :param path: Path to create
    :return: None
    """
    # define the name of the directory to be created

    try:
        os.makedirs(path)
    except OSError:
        print (f"Creation of the directory {path} failed")

def is_dir(path):
    """
    Check if the supplied path exists and if so, that it is a directory.

    Example:
        path = "/tmp/year/month/week/day"
        is_dir(path)

    :param path: Path to check
    :return: Boolean
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        else:
            raise ValueError(f"{path} is not a valid path.")
    else:
        return False

def persist_pipeline(pipeline, save_path):
    """
    Persist a pipeline.
    """
    joblib.dump(pipeline, save_path)

def load_pipeline(pipeline_path):
    """
    Load a saved pipeline.
    """
    joblib.load(pipeline_path)




