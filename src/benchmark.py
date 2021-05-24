from sklearn.datasets import fetch_20newsgroups
from sklearn import linear_model
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Get the data
categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

print("%d documents" % len(newsgroups_train.filenames))
print("%d categories" % len(newsgroups_train.target_names))

# Get features on the documents 
vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                n_features=20)
X_train = vectorizer.transform(newsgroups_train.data)

# Train the model using the training sets
regr = linear_model.LinearRegression()
regr.fit(X_train, newsgroups_train.target)

# Predict and check results 
documents_regressor = regr.predict(X_train)
documents_classifier = np.round(documents_regressor)

mse_results = mean_squared_error(newsgroups_train.target, documents_classifier)
r2_results = r2_score(newsgroups_train.target, documents_classifier)
print(f"MSE of prediction is - {mse_results:.4f}")
print(f"r2 score of prediction is - {r2_results:.4f}")