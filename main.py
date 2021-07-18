from fastapi import FastAPI, Body
from joblib import load
from model import inference
from data import process_data
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import os
import numpy as np
app = FastAPI()
import logging

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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
    age: int = Field(example=50)
    workclass: str = Field(example="Self-emp-not-inc")
    flngt: int = Field(example=83311)
    education: str = Field(example='Bachelors')
    education_num: int = Field(example=13)
    marital_status: str = Field(example='Married-civ-spouse')
    occupation: str = Field(example='Exec-managerial')
    relationship: str = Field(example='Husband')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=13)
    native_country: str = Field(example='United-States')

    class Config:
        # to change the col names into the formated params
        alias_generator = to_format
        allow_population_by_field_name = True
        
        

@app.get("/")
async def welcome():
    return {"msg": f"Hello there!"}

'''
= Body(default= True,
    examples={
        "normal2": {
            "summary": "A normal2 example",
            "description": "A **normal** item works correctly.",
            "value": {
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
        },
        "converted": {
            "summary": "An example with converted data",
            "description": "FastAPI can convert price `strings` to actual `numbers` automatically",
            "value": {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "flngt": 209642,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States"
            },
        }
    },
'''
@app.post("/predict")
def inference( X: features):
    
    data_dict = X.dict()
    
    model = load('model.joblib')
    encoder = load('encoder.joblib')
    lb = load('lb.joblib')
    data = pd.DataFrame(data_dict, index=[0])
    X_test, y_test, encoder,lb = process_data(
    data, categorical_features=cat_features, training=False, encoder= encoder, lb= lb
    )
    pred = model.predict(X_test)
    if pred == 0:
        return {"Income": "<=50k"}
    elif pred == 1:
        return {"Income": ">50k"}
