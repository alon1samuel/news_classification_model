from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def get_20newsgroups_data():
    # Get the data
    categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
    newsgroups_train = fetch_20newsgroups(
        subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    print("%d documents" % len(newsgroups_train.filenames))
    print("%d categories" % len(newsgroups_train.target_names))
    return newsgroups_train, newsgroups_test


def vectorize_text(text_train, text_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', max_features=1000)
    vectorizer.fit(text_train)
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    return X_train, X_test


def save_data_csv(X_train, y_train, X_test, y_test):
    df = pd.DataFrame.sparse.from_spmatrix(X_train)
    df['label'] = y_train
    df.to_csv('data/train.csv', index=False)
    df = pd.DataFrame.sparse.from_spmatrix(X_test)
    df['label'] = y_test
    df.to_csv('data/test.csv', index=False)


def main():
    newsgroups_train, newsgroups_test = get_20newsgroups_data()

    X_train, X_test = vectorize_text(
        newsgroups_train.data, newsgroups_test.data)
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    save_data_csv(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
