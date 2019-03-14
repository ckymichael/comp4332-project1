import numpy as np
import pandas as pd

import spacy
import re
from string import punctuation
from nltk.corpus import stopwords

import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2 as L2
from keras.utils import to_categorical

nlp = spacy.load('en_core_web_lg')
nlp_vec = spacy.load('en_vectors_web_lg')
stop_words = set(stopwords.words('english') + list(punctuation) + ['-PRON-'])


# Helper function to clean the review text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text, re.I | re.A).lower().strip()
    text = re.sub(' +', ' ', text)
    lemmatized = list()
    for token in text:
        lemma = token.lemma_
        if lemma not in stop_words and not lemma.isnumeric():
            lemmatized.append(''.join(lemma.split()))

    return " ".join(lemmatized)


# TODO Import data
def read_data():
    df_train = pd.read_csv('../data/train.csv')
    df_train.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    df_valid = pd.read_csv('../data/valid.csv')
    df_valid.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    df_test = pd.read_csv('../data/test.csv')
    df_test.drop(['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id'], axis=1, inplace=True)

    return df_train, df_valid, df_test


# TODO Complete feature engineering
def load_data():
    df_train, df_valid, df_test = read_data()
    return


# TODO Complete the composite LSTM --> CNN model

input_size = 0
embedding_size = 0
input_len = 300
hidden_size = 0
output_size = 5  # TODO Update to dynamic value

out_filters = 0
kernel_size = 0
padding = 'valid'
strides = 1
pool_size = 2

dropout_rate = 0.2
learning_rate = 0.01
batch_size = 128
total_epoch = 10

activation_def = 'relu'
optimizer_def = Adam()
regularizer_def = L2(0.001)

# New model
model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_len, trainable=False))

# LSTM Layer
model.add(LSTM(units=hidden_size))

# 2 1-D Convolution Stage
model.add(Conv1D(filters=out_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation_def, activity_regularizer=L2))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters=out_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation_def, activity_regularizer=L2))
model.add(MaxPooling1D(pool_size=pool_size))

# Dropout Layer
model.add(Dropout(rate=dropout_rate))

# Output
model.add(Dense(output_size, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

# Training
model.fit(epochs=total_epoch, batch_size=batch_size)

# TODO Complete model evaluation
# Testing
