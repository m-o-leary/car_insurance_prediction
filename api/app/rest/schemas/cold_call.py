from pydantic import BaseModel, validator, Field
from typing import Dict, FrozenSet, List
from enum import Enum, IntEnum
from typing import List, Optional, Tuple
from datetime import date
import math 

class JobEnum(str, Enum):
    """
    List of available jobs to send

    """
    management=""
    blue_collar='blue-collar'
    student='student'
    technician='technician'
    admin='admin.'
    services='services'
    self_employed='self-employed'
    retired='retired'
    housemaid='housemaid'
    entrepreneur='entrepreneur'
    unemployed='unemployed'
    other='other'

class MaritalEnum(str, Enum):
    """
    Marital status options
    """
    single='single'
    married='married'
    divorced='divorced'

class EducationEnum(str, Enum):
    """
    Education type options
    """
    primary='primary'
    secondary='secondary'
    tertiary='tertiary'

class CommunicationEnum(str, Enum):
    """
    Communication type options
    """
    telephone='telephone'
    cellular='cellular'
    other="other"

class MonthEnum(str, Enum):
    """
    Months options
    """
    jan='jan'
    feb='feb'
    mar='mar'
    apr='apr'
    may='may'
    jun='jun'
    jul='jul'
    aug='aug'
    sep='sep'
    oct='oct'
    nov='nov'
    dec='dec'

class OutcomeEnum(str, Enum):
    """
    Outcome options
    """
    failure="failure"
    other="other"
    success="success"

class Call(BaseModel):
    """
    Cold Call model
    """
    id: int = Field(None, description="The ID of the row in the data.")
    age: int = Field(None, description="The Age of the user.", gt=0, lt=120)
    job: JobEnum = Field(JobEnum.other, description="The job type of the customer.")
    marital: MaritalEnum = Field(None, description="The Marital status of the customer.")
    education: EducationEnum = Field(None, description="The education level of the customer.")
    default: int = Field(None, description="Whether the customer is in default or not.")
    balance: float = Field(None, description="The account balance of the customer.")
    has_home_insurance: int = Field(None, description="Has the customer home insurance.")
    car_loan: int = Field(None, description="Has the customer a car loan.")
    communication: CommunicationEnum = Field(CommunicationEnum.other, description="The communication type with the customer.")
    last_contact_day: int = Field(None, description="The day of the month of the previous contact with the customer.")
    last_contact_month: MonthEnum = Field(None, description="The month (3 letter abbreviation) of the previous contact with the customer.")
    num_contacts: int = Field(None, description="The number of contacts with the customer this campaign.")
    days_passed: int = Field(None, description="The number of days passed since previous contact with the customer.")
    previous_attempts: int = Field(None, description="The number of previous attempts to sell car insurance to this customer.")
    outcome: OutcomeEnum = Field(None, description="The outcome of previous attempts to sell car insurance to this customer.")
    call_start: str = Field(None, description="The time the call started. Format is HH:mm:ss", regex='^[0-9]{2}:[0-9]{2}:[0-9]{2}')
    call_end: str = Field(None, description="The time the call ended. Format is HH:mm:ss", regex='^[0-9]{2}:[0-9]{2}:[0-9]{2}')
    
    class Config:
        orm_mode = True
        anystr_strip_whitespace = True
        use_enum_values = True

class CallPrediction(Call):
    """
    Prediction model including the output
    """
    explanation: str= Field("", description="Explantion of model prediction.")
    prediction: int = Field(None, description="The model prediction. 1 = successful, 0 = unsuccessful.")
    prediction_proba: float = Field(0.5, description="""
        The model prediction probability. 
        Assuming a calibrated model, this is the output (between 0 and 1) of the model.
        Closer to 0 means very little chance of success and closer to 1 means very good chances of success.
        """)
    model_id: str = Field("", description="The id of the model used to generate this prediction.")
    
    class Config:
        orm_mode = True

class CallPredictionSaved(CallPrediction):
    """
    Prediction model including the output
    """
    pred_id: int
    
    class Config:
        orm_mode = True

class Model(BaseModel):
    """
    Machine learning classification model model :) .

    """
    estimator_id:str = Field(None, description="The unique Hash of the model pipeline, raw data, processed data hashes. If anything in either of those 3 elements change, so will this.")
    name: str = Field('No Name', description="The name of the model e.g. XGBOOST")
    params: dict = Field({}, decsiption="Parameter dict used by this model.")
    f1_score: float = Field(0.0, description="The f1 score for the classifier")
    accuracy: float = Field(0.0, description="The accuracy for the classifier")
    d: date = Field(..., description="Date of model training")
    raw_hash: str = Field(None, description="Unique hash of the pandas dataframe used as raw data." )
    processed_hash: str = Field(None, description="Unique hash of the processed dataframe. This changing indicates a difference in pre-processing pipeline definition or different raw data.")
    pipeline_hash: str = Field(None, description="Unique hash of the pre-processing pipeline. This changing indicates a difference in the pre-processing pipeline definition only.")
    predictions: List[CallPredictionSaved] = []
    class Config:
        orm_mode = True
    