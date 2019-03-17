import numpy as np
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch.optim as optim
import torch
from torchtext import data
from string import punctuation

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
nlp = spacy.load('en_core_web_lg')
nlp_vec = spacy.load('en_vectors_web_lg')
stop_words = set(stopwords.words('english') + list(punctuation) + ['-PRON-'])

# Helper function to clean the review text
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

    train_data_matrix = df_train['clean'].apply(get_word_vec).values
    train_data_matrix = np.vstack(train_data_matrix).reshape(train_data_matrix.shape[0], 150, 300)
    valid_data_matrix = df_valid['clean'].apply(get_word_vec).values
    valid_data_matrix = np.vstack(valid_data_matrix).reshape(valid_data_matrix.shape[0], 150, 300)
    test_data_matrix = df_test['clean'].apply(get_word_vec).values
    test_data_matrix = np.vstack(test_data_matrix).reshape(test_data_matrix.shape[0], 150, 300)

    K = 5
    train_data_label = to_categorical(df_train['stars'], num_classes=K)
    valid_data_label = to_categorical(df_valid['stars'], num_classes=K)

    return train_data_matrix, train_data_label, valid_data_matrix, valid_data_label, test_data_matrix, K


# TODO Complete the composite LSTM --> CNN model
train_data_matrix, train_data_label, valid_data_matrix, valid_data_label, test_data_matrix, output_size = load_data()

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data_matrix, valid_data_matrix, test_data_matrix),
    batch_size=BATCH_SIZE,
    device=device)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    print(
        f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')