from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)

salary_lessthan50_sample =  {
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
            }

salary_morethan50_sample =  {
                "age": 43,
                "workclass": "Federal-gov",
                "flngt": 410867,
                "education": "Doctorate",
                "education-num": 16,
                "marital-status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": "United-States"
            }

def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"msg": f"Hello there!"}


def test_post_path_1():
    input_dict = salary_lessthan50_sample
    r = client.post("/predict", json=input_dict)
    assert r.status_code == 200
    assert json.loads(r.text)["Income"] == "<=50k"

def test_post_path_2():
    input_dict = salary_morethan50_sample
    r = client.post("/predict", json=input_dict)
    assert r.status_code == 200
    assert json.loads(r.text)["Income"] == "<=50k"

