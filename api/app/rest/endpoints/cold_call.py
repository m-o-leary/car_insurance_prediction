import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, validator, ValidationError
from typing import Dict, FrozenSet, List
from app.rest.schemas.cold_call import Call, Model, CallPrediction, CallPredictionSaved
from app.rest.dependencies import get_db
from app.rest.database import crud
from sqlalchemy.orm import Session

# Instantiate the router
router = APIRouter( prefix="/insurance_model" )

@router.get("/models/", response_model=List[Model], status_code=200, tags=["Model"])
async def get_model_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return a listt of models.
    """
    models = crud.get_models(db, skip=skip, limit=limit)
    for m in models:
        m.params = json.loads(m.params)
    return models

@router.get("/models/{model_id}", status_code=200, tags=["Model"])
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Retrieve specific model.

    """

    model = crud.get_model(d, model_id=model_id)
    model.params = json.loads(model.params)
    return model

@router.post("/models/", response_model=Model, status_code=201, tags=["Model"])
async def add_model_to_database(model: Model, db: Session = Depends(get_db)):
    """
    Add a model to the database.

    """
    model = crud.create_model(db=db, model=model)
    model.params = json.loads(model.params)
    return model

@router.get("/predictions/", response_model=List[CallPredictionSaved], status_code=200, tags=["Model"])
async def get_model_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return a list of predictions.
    """
    models = crud.get_predictions(db, skip=skip, limit=limit)
    return models

@router.get("/predictions/{pred_id}", response_model=List[CallPredictionSaved], status_code=200, tags=["Model"])
async def get_model(pred_id: str, db: Session = Depends(get_db)):
    """
    Retrieve specific prediction.

    """

    pred = crud.get_prediction(db, pred_id=pred_id)

    return pred

@router.post("/predictions/", response_model=CallPredictionSaved, status_code=201, tags=["Model"])
async def add_model_to_database(pred: Call, db: Session = Depends(get_db)):
    """
    Request a prediction from a cold call details.

    """

    # run pred
    #  ... 
    pred = CallPrediction(**pred.dict(), prediction=1, prediction_proba = 0.89)

    return crud.create_prediction_row(db=db, pred=pred)