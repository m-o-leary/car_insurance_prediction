from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, Date
from sqlalchemy.orm import relationship

from .database import Base


class Model(Base):
    __tablename__ = "models"
    
    estimator_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    params = Column(String)
    f1_score = Column(Float)
    accuracy = Column(Float)
    d = Column(Date)
    raw_hash = Column(String)
    processed_hash = Column(String)
    pipeline_hash = Column(String)

    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    __tablename__ = "predictions"
    
    pred_id = Column(Integer, primary_key=True, index=True)
    id = Column(Integer)
    age = Column(Integer)
    job = Column(String)
    marital = Column(String)
    education = Column(String)
    default = Column(Boolean)
    balance = Column(Float)
    has_home_insurance = Column(Boolean)
    car_loan = Column(Boolean)
    communication = Column(String)
    last_contact_day = Column(Integer)
    last_contact_month = Column(String)
    num_contacts = Column(Integer)
    days_passed = Column(Integer)
    previous_attempts = Column(Integer)
    outcome = Column(String)
    call_start = Column(String)
    call_end = Column(String)
    prediction = Column(Boolean)
    prediction_proba = Column(Float)
    explanation = Column(String,default="")
    model_id = Column(String, ForeignKey("models.estimator_id"))

    model = relationship("Model", back_populates="predictions")