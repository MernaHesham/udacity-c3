# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from data import process_data
from model import train_model, slice_metrics
from joblib import dump, load

# Add code to load in the data.
data = pd.read_csv('data/census_modified.csv', index_col=0)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
print('train')
print(train)

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
print("***********************")
# Proces the test data with the process_data function.
X_test, y_test, encoder,lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb= lb
)

print('len X_train')
print(len(X_train[0]))
print('y_train')
print(y_train)

print('encoder')
print(encoder)

# Train and save a model.
model = train_model(X_train, y_train)
print(model)

dump(model, 'model.joblib') # save the model
dump(encoder, 'encoder.joblib') # save the model

slice_metrics(test, cat_features, model, encoder, lb)