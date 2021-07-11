from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from data import process_data
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def slice_metrics(test_data, cat_features, model, encoder, lb):
    print("_______slice____")
    print('test data')
    print(test_data)
    print('feature: ', cat_features[1])

    with open('slice_output.txt', 'w') as f:
        for cat in range(len(cat_features)):
            f.write('\n for {} \n'.format(cat_features[cat]))

            for cls in test_data[cat_features[cat]].unique():
                print("for ", cls)

                df_feature = test_data[test_data[cat_features[cat]]==cls]
                print('sliced df: ')
                print(df_feature)

                X_test, y_test, encoder,lb = process_data(
                df_feature, categorical_features=cat_features, label="salary", training=False, encoder= encoder, lb= lb
                )

                preds = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, preds)
                print("for {} precision is {} and recall is {}".format(cls, precision, recall))
                f.write("for {} precision is {} and recall is {} \n".format(cls, precision, recall))
                print("-------------")


    
    return 1