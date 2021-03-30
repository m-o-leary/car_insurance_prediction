from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.rest.database import crud, models, database
from app.rest.database.database import SessionLocal, engine

# Init DB
models.Base.metadata.create_all(bind=engine)

# Tags for the different routes
tags_metadata = [
    {
        "name": "Model",
        "description": "Operations to get model outputs relating to cold call model.",
    }
]

# # Spin up the app
app = FastAPI(
    title="Car Insurance Cold Calling Sales Prediction API",
    description="This is an API to provide predictions from the CarInsurancePredictionModel[TM]",
    version="0.0.1",
    openapi_tags=tags_metadata)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.rest.endpoints.cold_call import router as cold_call_router

app.include_router(cold_call_router)

