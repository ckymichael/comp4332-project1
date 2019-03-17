import numpy as np
import pandas as pd

import spacy
import re
from string import punctuation
from nltk.corpus import stopwords

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2 as L2

import json

nlp = spacy.load('en_core_web_lg')
nlp_vec = spacy.load('en_vectors_web_lg')
stop_words = set(stopwords.words('english') + list(punctuation) + ['-PRON-'])


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I | re.A).lower().replace('\n', '').strip()
    text = re.sub(' +', ' ', text)
    text = nlp(text)
    lemmatized = list()
    for token in text:
        lemma = token.lemma_
        if lemma not in stop_words and not lemma.isnumeric():
            lemmatized.append(''.join(lemma.split()))

    return " ".join(lemmatized)


def get_word_vec(text):
    seq = np.array([nlp_vec.vocab.get_vector(word) for word in text.split() if nlp_vec.vocab.has_vector(word)])
    if seq.size > 0:
        seq = pad_seq(seq)
    else:
        seq = np.zeros((150, 300))

    return seq


def pad_seq(seq):
    return pad_sequences(seq.transpose(), dtype='float32', maxlen=150).transpose()


def read_data():
    df_train = pd.read_csv('../data/train.csv')
    df_train.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    df_valid = pd.read_csv('../data/valid.csv')
    df_valid.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    df_test = pd.read_csv('../data/test.csv')
    df_test.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    return df_train, df_valid, df_test


def load_data():
    df_train, df_valid, df_test = read_data()

    df_train['text'] = df_train['text'].apply(clean_text)
    df_valid['text'] = df_valid['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)

    df_valid['text'] = df_valid['text'].astype('str')
    df_train['text'] = df_train['text'].astype('str')
    df_test['text'] = df_test['text'].astype('str')

    train_data_matrix = df_train['text'].apply(get_word_vec).values
    train_data_matrix = np.vstack(train_data_matrix).reshape(train_data_matrix.shape[0], 150, 300)
    valid_data_matrix = df_valid['text'].apply(get_word_vec).values
    valid_data_matrix = np.vstack(valid_data_matrix).reshape(valid_data_matrix.shape[0], 150, 300)
    test_data_matrix = df_test['text'].apply(get_word_vec).values
    test_data_matrix = np.vstack(test_data_matrix).reshape(test_data_matrix.shape[0], 150, 300)

    K = 5
    train_data_label = to_categorical(df_train['stars'], num_classes=K)
    valid_data_label = to_categorical(df_valid['stars'], num_classes=K)

    return train_data_matrix, train_data_label, valid_data_matrix, valid_data_label, test_data_matrix, K


train_data_matrix, train_data_label, valid_data_matrix, valid_data_label, test_data_matrix, output_size = load_data()

embedding_size = 300
time_steps = 150

dropout_rate = 0.5
learning_rate = 0.01
batch_size = 64
total_epoch = 20

activation_def = 'relu'
optimizer_def = Adam()
regularizer_def = L2(0.0001)

# New model
model = Sequential()

# Double LSTM
model.add(LSTM(embedding_size, input_shape=(time_steps, embedding_size), return_sequences=True, kernel_regularizer=regularizer_def))
model.add(LSTM(embedding_size, input_shape=(time_steps, embedding_size), kernel_regularizer=regularizer_def))

# Dropout Layer
model.add(Dropout(rate=dropout_rate))

# Output
model.add(Dense(output_size, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
print(model.summary())

# Training
train_history = model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size, validation_data=(valid_data_matrix, valid_data_label))

model.save('lstm_'+str(total_epoch)+'epoch.h5')
with open('./lstm_history.json', 'w') as fp:
    json.dump(train_history.history, fp)

# Evaluation
train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))

valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))
