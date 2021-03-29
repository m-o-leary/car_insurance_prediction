DATA_URL="https://storage.googleapis.com/kaggle-data-sets/1411/2534/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210327%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210327T123106Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=08244af6987a27e69312cd7e9c67529b5522279cee8f385a79c44e95a6bd9e708451b371384fc35376bc1ff8250bd1fd6b2595ad824ce728e08485ae8021158ff671a7c6df6c0940a58a68379f11734390dd68afea6099a206312fe3c10d880d02b1decd4c3a33539bff730c16ae365ac4823b1d53e6e214eee00b613230468e54b16789903acd9755bdaa0681a0609df43fea8bc8c979b4af90a006c955d8e7d507a958cbef7f84a8e653abec0ae697dc5b35f4cf98f243b8ccbe17c64279581e60941b90e48b31f9432d1b6fcb3f3000a3fccbb5dc1da6f32f98a4148bce3f778e6385b59ad48827f5ac6dc5c96bb8ef9b7191f8993b905a3bc0a92560364f"
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