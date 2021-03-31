import json
import sqlalchemy
import pandas as pd
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

    model = crud.get_model(d, model_id=model_id)
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

@router.post("/predictions/{model_id}", response_model=CallPredictionSaved, status_code=201, tags=["Model"])
async def run_prediction(model_id, pred: Call, db: Session = Depends(get_db), clf = Depends(model_prediction.load_pipeline)):
    """
    Submit cold call details for success prediction.
    The result and the details will be stored in the database.

    """


    # run pred
    #  ... 
    pred = CallPrediction(**pred.dict(), prediction=1, prediction_proba = 0.89)

    return crud.create_prediction_row(db=db, pred=pred)