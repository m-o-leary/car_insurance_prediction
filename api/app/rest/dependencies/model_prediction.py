import joblib
import pandas as pd
def load_pipeline(model_id):
    """
    Load a model.
    
    Full pipeline training and having fit_transform / predict / predict_proba methods

    Models are expected to be located in the /models directory

    Parameters
    ----------
    model_id : Str
        Model hash id to load.
    """

    pipeline = joblib.load(f"/models/{model_id}.pkl")

    return pipeline

def map_call_line_to_prediction(call):
    """
    Load a model.
    
    Full pipeline training and having fit_transform / predict / predict_proba methods

    Models are expected to be located in the /models directory

    Parameters
    ----------
    model_id : Str
        Model hash id to load.
    """

    return pd.DataFrame({
        'Id': [call.id], 
        'Age':[call.age], 
        'Job': [call.job], 
        'Marital': [call.marital], 
        'Education':[call.education], 
        'Default': [call.default], 
        'Balance': [call.balance],
        'HHInsurance': [call.has_home_insurance], 
        'CarLoan': [call.car_loan], 
        'Communication': [call.communication], 
        'LastContactDay': [call.last_contact_day],
        'LastContactMonth': [call.last_contact_month], 
        'NoOfContacts': [call.num_contacts], 
        'DaysPassed': [call.days_passed], 
        'PrevAttempts': [call.previous_attempts],
        'Outcome': [call.outcome], 
        'CallStart': [call.call_start], 
        'CallEnd': [call.call_end]
    })
