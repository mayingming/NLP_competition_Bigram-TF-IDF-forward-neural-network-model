import json
import nltk
import re
import pickle
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Lambda
from keras.models import Model
from bert_embedding import BertEmbedding


lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
stopwords.update(["et", "al", "mr"])


def remove(text):
    remove_chars = '[0-9’"#$%&\'()*+,-?!？！./:;<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)


def lemmatize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    lemmatized_sentence = []
    for word in tokens:
        if word not in stopwords:
            if word == "co":
                lemmatized_sentence.append("carbon")
            else:
                lemmatized_sentence.append(word)
    return " ".join(lemmatized_sentence)


def preprocess(events, dataset):
    for event in events:
        paragragh = lemmatize_sentence(remove(event))
        dataset.append(paragragh)


fname_train = 'train.json'
fname_dev = 'dev.json'
fname_test = 'test-unlabelled.json'
fname_external = 'external_train_data.json'

result_dev = {}
result_test = {}
misinformation = []
external = []
dev_data = []
test_data = []
y_external = []
y_dev = []

with open(fname_train) as json_file:
    data = json.load(json_file)
    for event in data:
        misinformation.append(data[event]['text'].lower())

with open(fname_external) as json_file:
    data = json.load(json_file)
    for event in data:
        external.append(data[event]['text'].lower())
        y_external.append(data[event]['label'])

with open(fname_dev) as json_file:
    data = json.load(json_file)
    for event in data:
        dev_data.append(data[event]['text'].lower())
        y_dev.append(data[event]['label'])
        result_dev.update({event: {'label': 0}})

with open(fname_test) as json_file:
    data = json.load(json_file)
    for event in data:
        test_data.append(data[event]['text'].lower())
        result_test.update({event: {'label': 0}})

ymis = []
for i in range(len(misinformation)):
    ymis.append(1)

print("Number of misinformation events =", len(misinformation))
print("Number of external events =", len(external))
print("Number of dev events =", len(dev_data))
print("Number of test events =", len(test_data))

train_data, y_train = misinformation + external, ymis + y_external
X_dev = []
X_test = []
X_train = []
X_mis = []

preprocess(train_data, X_train)
preprocess(dev_data, X_dev)
preprocess(test_data, X_test)
print("Preprocess finished")

model = SentenceTransformer('bert-large-nli-mean-tokens')

bert_x_train = model.encode(X_train)
vocab_size = len(bert_x_train[0])
print(vocab_size)
bert_x_train = np.array(bert_x_train)
bert_x_dev = model.encode(X_dev)
vocab_size_dev = len(bert_x_dev[0])
print(vocab_size_dev)
bert_x_dev = np.array(bert_x_dev)
bert_x_test = model.encode(X_test)
bert_x_test = np.array(bert_x_test)
y_train = np.array(y_train)
y_dev = np.array(y_dev)

bert_x_train = bert_x_train.reshape((bert_x_train.shape[0], 1, bert_x_train.shape[1]))
bert_x_dev = bert_x_dev.reshape((bert_x_dev.shape[0], 1, bert_x_dev.shape[1]))
bert_x_test = bert_x_test.reshape((bert_x_test.shape[0], 1, bert_x_test.shape[1]))

print("Data ready to train")

# save bert_train_new
pickle_out = open("bert_train_1024.pickle", "wb")
pickle.dump(bert_x_train, pickle_out)
pickle_out.close()

# save bert_dev_new
pickle_out = open("bert_dev_1024.pickle", "wb")
pickle.dump(bert_x_dev, pickle_out)
pickle_out.close()

# save bert_test_new
pickle_out = open("bert_test_1024.pickle", "wb")
pickle.dump(bert_x_test, pickle_out)
pickle_out.close()

# load bert_train
# pickle_in = open("bert_train_1024.pickle", "rb")
# bert_x_train = pickle.load(pickle_in)

# load bert_dev
# pickle_in = open("bert_dev_1024.pickle", "rb")
# bert_x_dev = pickle.load(pickle_in)
#
# # load bert_test
# pickle_in = open("bert_test_1024.pickle", "rb")
# bert_x_test = pickle.load(pickle_in)


input_layer = Input((None, 1024))
model = Dense(256, activation='relu')(input_layer)
model = Bidirectional(LSTM(units=512, return_sequences=True))(model)
model = Bidirectional(LSTM(64, return_sequences=False))(model)
pred = Dense(1, activation='sigmoid')(model)
model = Model(inputs=input_layer, outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(bert_x_train, y_train, epochs=10, verbose=True, validation_data=(bert_x_dev, y_dev), batch_size=32)
predictions = model.predict(bert_x_dev, verbose=True, batch_size=32)


# Dev predict
for i in range(0, len(predictions)):
    if float(predictions[i]) > 0.5:
        result_dev.update({'dev-%d' % i: {'label': 1}})
    else:
        result_dev.update({'dev-%d' % i: {'label': 0}})
with open("dev_result.json", "w") as ftow:
    json.dump(result_dev, ftow)

# Test predict
# for i in range(0, len(predictions)):
#     print(float(predictions[i]))
#     if float(predictions[i]) > 0.5:
#         result_test.update({'test-%d' % i: {'label': 1}})
#     else:
#         result_dev.update({'dev-%d' % i: {'label': 0}})
#
# with open("test-output.json", "w") as ftow:
#     json.dump(result_test, ftow)

print("Done")
