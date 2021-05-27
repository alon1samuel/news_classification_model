from src import data_handler
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
# Framework taken from - https://realpython.com/python-keras-text-classification/#further-reading

newsgroups_train, newsgroups_test = data_handler.get_20newsgroups_data()

vectorizer = CountVectorizer()
vectorizer.fit(newsgroups_train.data)
X_train = vectorizer.transform(newsgroups_train.data)
X_test  = vectorizer.transform(newsgroups_test.data)


input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, newsgroups_train.target,
                    epochs=5,
                    validation_data=(X_test, newsgroups_test.target),
                    batch_size=10)

from keras.backend import clear_session
clear_session()

loss, accuracy = model.evaluate(X_test, newsgroups_test.target, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)
