from pydantic import BaseModel, validator
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

class MaritalEnum(str: Enum):
    """
    Marital status options
    """
    single='single'
    married='married'
    divorced='divorced'

class EducationEnum(str: Enum):
    """
    Education type options
    """
    primary='primary'
    secondary='secondary'
    tertiary='tertiary'

class CommunicationEnum(str: Enum):
    """
    Communication type options
    """
    telephone='telephone'
    cellular='cellular'
    other="other"

class MonthEnum(str: Enum):
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

class OutcomeEnum(str: Enum):
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
    id: int = None
    age: int = None
    job: JobEnum = JobEnum.other
    marital: MaritalEnum = None
    education: EducationEnum = None
    default: int = None
    balance: float = None
    has_home_insurance: int = None
    car_loan: int = None
    communication: CommunicationEnum = CommunicationEnum.other
    last_contact_day: int = None
    last_contact_month: MonthEnum = None
    num_contacts: int = None
    days_passed: int = None
    previous_attempts: int = None
    outcome: OutcomeEnum = None
    call_start: str = None
    call_end: str = None

class Model(BaseModel):
    """
    Machine learning model data model
    """
    id: str
    name: str
    params: dict
    score: float
    date: date
    