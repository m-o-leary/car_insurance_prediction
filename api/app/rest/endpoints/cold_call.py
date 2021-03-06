import json
import sqlalchemy
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, validator, ValidationError
from typing import Dict, FrozenSet, List
from app.rest.schemas.cold_call import Call, Model, CallPrediction, CallPredictionSaved
from app.rest.dependencies import get_db
from app.rest.dependencies import model_prediction
from app.rest.database import crud
from sqlalchemy.orm import Session

# Instantiate the router
router = APIRouter( prefix="/insurance_model" )

@router.get("/models/", response_model=List[Model], status_code=200, tags=["Model"])
async def get_model_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return a list of model.
    """
    models = crud.get_models(db, skip=skip, limit=limit)
    for m in models:
        m.params = json.loads(m.params)
    return models

@router.get("/models/{model_id}", status_code=200, tags=["Model"])
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Retrieve specific model details.

    """

    model = crud.get_model(db, model_id=model_id)
    model.params = json.loads(model.params)
    return model

@router.post("/models/", response_model=Model, status_code=201, tags=["Model"])
async def add_model_to_database(model: Model, db: Session = Depends(get_db)):
    """
    Add model run details to the database.

    """
    try:
        model = crud.create_model(db=db, model=model)
        model.params = json.loads(model.params)
        return model
    except sqlalchemy.exc.IntegrityError:
        raise HTTPException(status_code=409, detail="Model with that hash already exists!")  

@router.get("/predictions/", response_model=List[CallPredictionSaved], status_code=200, tags=["Model"])
async def get_prediction_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return a list of predictions.
    """
    predictions = crud.get_predictions(db, skip=skip, limit=limit)
    return predictions

@router.get("/predictions/{pred_id}", response_model=List[CallPredictionSaved], status_code=200, tags=["Model"])
async def get_prediction(pred_id: str, db: Session = Depends(get_db)):
    """
    Retrieve specific prediction details.

    """

    pred = crud.get_prediction(db, pred_id=pred_id)

    return pred

@router.post("/predictions/{model_id}", response_model=List[CallPredictionSaved], status_code=201, tags=["Model"])
async def run_prediction(model_id, calls: List[Call], db: Session = Depends(get_db)):
    """
    Submit cold call details for success prediction.
    The result and the details will be stored in the database.

        Business rules 

    """
    try:
        clf = model_prediction.load_pipeline(model_id)
        call_df = model_prediction.map_call_line_to_prediction(calls)
        preds__ = clf.predict_proba(call_df)

        predictions = []
        
        for i, pred__ in enumerate(preds__):
            explanation = model_prediction.business_decision(pred__, calls[i])
            pred = CallPrediction(**calls[i].dict(), 
                explanation=explanation,
                prediction=int(pred__[1] > 0.5), 
                prediction_proba = pred__[1])
            pred.model_id = model_id
            predictions.append(crud.create_prediction_row(db=db, pred=pred))
        return predictions

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No cached model with id:{model_id} could be found!")