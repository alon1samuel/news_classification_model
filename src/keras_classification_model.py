import numpy as np
import keras
from keras.preprocessing import sequence, text
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from src import data_handler
# Framework by - https://github.com/CSCfi/machine-learning-scripts/blob/master/examples/tf2-20ng-cnn.py


def build_nn_network(input_dims):
    # Model inspired by - https://www.kaggle.com/ashoksrinivas/nlp-with-tfidf-neural-networks
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(128, kernel_initializer='glorot_uniform', input_dim=input_dims))

    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid' ))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    # loss = keras.losses.sparse_categorical_crossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['acc'])


    print(model.summary())

    return model



model_callback = keras.callbacks.EarlyStopping(monitor='acc', mode='auto') 

newsgroups_train, newsgroups_test = data_handler.get_20newsgroups_data()


# Vectorize the text samples into a 2D integer tensor.

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000 

tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(newsgroups_train.data)
sequences = tokenizer.texts_to_sequences(newsgroups_train.data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(newsgroups_train.target))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# Prepare the embedding matrix:

print('Preparing embedding matrix.')
MAX_NUM_WORDS = 10000

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_dim = 100

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

input_dims = X_train.shape[1]
print(f"Input shape - {X_train_np.shape}")
nn_model = build_nn_network(input_dims)

nn_model.fit(X_train_np, newsgroups_train.target, batch_size=1900, epochs=10, validation_split=0.2)

test_pred = nn_model.predict(X_test_np)

print(f"Accuracy on test - {accuracy_score(test_pred, newsgroups_test.target)}")

print()