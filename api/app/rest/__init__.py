from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


from rest.endpoints import cold_call

app.include_router(cold_call.router)

