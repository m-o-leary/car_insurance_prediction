# Car Insurance Prediction

The point of this repo is to demonstrate an end to end example of a classification problem.

## Structure

There are 2 parts to this project:

1.  Data transformation and model training code in `car_insurance_pipeline/`
2.  Model serving over REST API code in `api/`

To run the 

### Data transformation and model training

The data is from this Kaggle competition https://www.kaggle.com/kondla/carinsurance. 

To download the data run the following:

```bash
chmod +x get_data.sh
./get_data.sh
```