import joblib
import pandas as pd
import numpy as np

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

def map_call_line_to_prediction(calls):
    """
    Load a model.
    
    Full pipeline training and having fit_transform / predict / predict_proba methods

    Models are expected to be located in the /models directory

    Parameters
    ----------
    model_id : Str
        Model hash id to load.
    """
    dfs = []
    for call in calls:
        dfs.append(pd.DataFrame({
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
        }))

    return pd.concat(dfs)

def business_decision(prediction, call):
    """
    Business logic to take a call details,
     model output and make a decision to return based on this business logic

    The logic is as follows:

    (The probability of success * the potential gain ) 
    x potential lifetime value multiplier for first time customer 
    - ( 1 - probability of succss * cost of calling )

    Args:
        prediction (np.ndarray): An array of predictions 
        where the first element is the probability of 
        an outcome of the first class (0) and 
        the second element is the probability of the secnd class (1).
        call (Call): Call object containing details on which prediction is made 

    Returns:
        str: And explanation / advice to whoever consumees the API.
    """
    cost_of_calling = 15
    gain_from_selling = 300
    threshold = 0.4
    
    ltv = 1 if call.outcome is "success" else 1.2

    net_gain = ltv * ( prediction[1] * gain_from_selling ) - ( prediction[0]* cost_of_calling )
    explanation = "Calling this customer is likely to result in them purchasing car insurance." \
        if prediction[1] > threshold\
        else "Calling this customer is unlikely to result in a successful sale."
    explanation += f" The expexcted value of this call is ${np.round(net_gain, 1)}"
    return explanation
