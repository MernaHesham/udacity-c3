import requests
import json

url = r'https://merna-demo-app.herokuapp.com/predict'

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

resp = requests.post(url, data=json.dumps(salary_lessthan50_sample))

print(resp.text)