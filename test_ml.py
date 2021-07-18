import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import compute_model_metrics, train_model, inference
from joblib import load



def test_train_model():
    data = pd.read_csv('census_modified.csv', index_col=0)
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert model != None


def test_compute_model_metrics():
    data = pd.read_csv('census_modified.csv', index_col=0)
    model = load('model.joblib')
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder,lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb= lb
    )
    preds = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert precision != 0
    assert recall != None


def test_inference():
    data = pd.read_csv('census_modified.csv', index_col=0)
    model = load('model.joblib')
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder,lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb= lb
    )
    preds = inference(model, X_test)
    assert preds.all() != None

