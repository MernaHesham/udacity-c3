from fastapi import FastAPI
from joblib import load
from model import inference
from data import process_data
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import numpy as np
app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

def to_format(string: str) -> str:
    return '-'.join(word for word in string.split('_'))

class features(BaseModel):
    age: int
    workclass: str
    flngt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str 

    class Config:
        # to change the col names into the formated params
        alias_generator = to_format
        schema_extra = {
            "example": {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "flngt": 83311,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 13,
                "native-country": "United-States"
            },
            
        }
        

@app.get("/")
async def welcome():
    return {"msg": f"Hello there!"}


@app.post("/predict")
def inference( X: features):
    data_dict = features
    model = load('model.joblib')
    encoder = load('encoder.joblib')
    lb = load('lb.joblib')
    data = pd.DataFrame.from_dict(features, orient='index')
    X_test, y_test, encoder,lb = process_data(
    data, categorical_features=cat_features, training=False, encoder= encoder, lb= lb
    )
    pred = inference(model, X_test)
    if pred == 0:
        return {"Income": "<=50k"}
    elif pred == 1:
        return {"Income": ">50k"}
