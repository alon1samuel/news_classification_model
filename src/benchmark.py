from sklearn.datasets import fetch_20newsgroups
from sklearn import linear_model
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Get the data
categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print("%d documents" % len(newsgroups_train.filenames))
print("%d categories" % len(newsgroups_train.target_names))

# Get features on the documents 
# vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
#                                 n_features=20)
# X_train = vectorizer.transform(newsgroups_train.data)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                    stop_words='english')
vectorizer.fit(newsgroups_train.data)
X_train = vectorizer.transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
    

def predict_classification(data, regressor):
    documents_regression = regressor.predict(data)
    documents_classification = np.round(documents_regression)
    return documents_classification

def check_results(data_classification, data_target):
    mse_results = mean_squared_error(data_target, data_classification)
    r2_results = r2_score(data_target, data_classification)
    print(f"MSE of prediction is - {mse_results:.4f}")
    print(f"r2 score of prediction is - {r2_results:.4f}")



# Train the model using the training sets
regr = linear_model.LinearRegression()
regr.fit(X_train, newsgroups_train.target)


# Predict and check results 
print(f"Results for train set")
train_classification = predict_classification(X_train, regr) 
check_results(train_classification, newsgroups_train.target)

print(f"Results for test set")
test_classification = predict_classification(X_test, regr) 
check_results(test_classification, newsgroups_test.target)
len(newsgroups_test.target)