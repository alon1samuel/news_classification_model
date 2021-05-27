# Framework using - https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#sphx-glr-auto-examples-ensemble-plot-ensemble-oob-py

from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import metrics
import tqdm

from src import data_handler
RANDOM_STATE = 123

# Get the data
newsgroups_train, newsgroups_test = data_handler.get_20newsgroups_data()

X_train, X_test = data_handler.vectorize_text(newsgroups_train.data, newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Build classifiers
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
accuracy_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


# Range of `n_estimators` values to explore.
min_estimators = 140
max_estimators = 200

best_acc = 0
best_cm = {}

for label, clf in ensemble_clfs:
    print(f"Working on - {label} classifier")
    for i in tqdm.tqdm(range(min_estimators, max_estimators + 1)):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)

        current_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(current_pred, y_test)
        if accuracy> best_acc:
            best_cm['cm'] = metrics.confusion_matrix(current_pred, y_test)
            best_cm['estimators_number'] = i
            best_cm['label'] = label
            best_cm['acc'] = accuracy

        accuracy_rate[label].append((i, accuracy))

print("Best model confusion matrix:")
for key in best_cm.keys():
    print(f"{key} - {best_cm[key]}")

# Plot accuracy of the models
plt.figure()
for label, clf_acc in accuracy_rate.items():
    xs, ys = zip(*clf_acc)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.show()
