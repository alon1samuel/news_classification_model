# This is the Logistic regression benchmark


from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import metrics
from src import data_handler
import numpy as np

# Relevant functions
def predict_classification(data, regressor):
    documents_regression = regressor.predict(data)
    documents_classification = np.round(documents_regression)
    return documents_classification

def check_results(data_classification, data_target):
    accuracy_res = accuracy_score(data_target, data_classification)
    print(f"Accuracy of prediction is - {accuracy_res:.4f}")


# Get data
newsgroups_train, newsgroups_test = data_handler.get_20newsgroups_data()

X_train, X_test = data_handler.vectorize_text(newsgroups_train.data, newsgroups_test.data)

# Train the model using the training sets
regr = linear_model.LogisticRegression()
regr.fit(X_train, newsgroups_train.target)


# Predict and check results 
print(f"Results for train set")
train_classification = predict_classification(X_train, regr) 
check_results(train_classification, newsgroups_train.target)

print(f"Results for test set")
test_classification = predict_classification(X_test, regr) 
check_results(test_classification, newsgroups_test.target)

print(f"Confusion matrix - ")
print(metrics.confusion_matrix(test_classification,newsgroups_test.target))