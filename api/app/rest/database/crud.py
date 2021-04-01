import json
from sqlalchemy.orm import Session

from app.rest.database import models
from app.rest.schemas import cold_call


def get_model(db: Session, model_id: str):
    return db.query(models.Model).filter(models.Model.estimator_id == model_id).first()

def get_models(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Model).offset(skip).limit(limit).all()

def create_model(db: Session, model: cold_call.Model):
    model.params = json.dumps(model.params)
    db_model = models.Model(**model.dict())
    db.add(db_model)
    db.commit()
    # Only needed if the DB adds data to model e.g. auto - incremented ID
    db.refresh(db_model)
    return db_model

def get_prediction(db: Session, pred_id: str):
    return db.query(models.Prediction).filter(models.Prediction.id == pred_id).all()

def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).offset(skip).limit(limit).all()

def create_prediction_row(db: Session, pred: cold_call.CallPrediction):
    db_pred = models.Prediction(**pred.dict())
    db.add(db_pred)
    db.commit()
    # Only needed if the DB adds data to model e.g. auto - incremented ID
    db.refresh(db_pred)
    return db_pred
