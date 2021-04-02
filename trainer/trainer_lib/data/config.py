DATA_URL="https://github.com/m-o-leary/car_insurance_data/raw/main/kaggle_data.zip"
TRAIN_DATA = "carInsurance_train.csv"
TEST_DATA = "carInsurance_test.csv"
TARGET_VARIABLE = "CarInsurance"
# Model columns
ONE_HOT_CATEGORICAL_COLUMNS = [
    "Job",
    "Marital",
    "Communication",
    "CallTimeOfDay"
]

SCALABLE_NUMERIC_COLUMNS = [
    "Age",
    "Balance",
]

NUMERIC_COLUMNS = [
    "LastContactDay",
    "LastContactMonth",
    "NoOfContacts",
    "DaysPassed",
    "PrevAttempts",
    "CallDurationMins",
    "Education"
]

BINARY_COLUMNS = [
    "Default",
    "HHInsurance",
    "CarLoan",
    "Outcome"
]
# All model columns
ALL_COLUMNS = ONE_HOT_CATEGORICAL_COLUMNS + SCALABLE_NUMERIC_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS