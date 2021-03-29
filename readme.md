# Car Insurance Prediction Problem

The goal of this project is to develop and serve a machine learning model that will predict whether a customer will buy car insurance or not.
## Project Structure

The overall priject structure is the following:

```text

.
├── api                         <- Code for REST API
├── data
│   ├── processed               <- Output from pre-processing steps
│   └── raw                     <- Raw data and documentation (Read only)
├── docker-compose.yml          
├── model                       <- Saved models in pickle format
├── notebooks                   <- Notebooks containing EDA and processing and model taining steps defined in the trainer_lib 
├── readme.md
├── reports                     <- Profiling outputs
└── trainer                     <- Source code for all pre-processing / model training / evaluation
    ├── trainer_lib
```

The project is broken into 2 parts:

1. Data processing and model training code in (code in `/trainer/`).
2. Model serving over REST API code in (code in `api/`).

## Running project

You will need Docker installed to run this project code.
( Details available here https://docs.docker.com/get-docker/ )

To start and run the entire project locally (assuming you have docker installed):

```sh
git clone https://github.com/buddythumbs/car_insurance_prediction.git
cd car_insurance_prediction
docker-compose up
```

This will build all the required images and start the containers.
### EDA / Pre-processing and Model training

The easiest way to work through the end to end model training process is a jupyter lab instance running in the `trainer_1` container.

To access this go to http://127.0.0.1:8888/lab?token=justatokengesture in a decent web browser.

### Model predictions
