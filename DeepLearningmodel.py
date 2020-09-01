import json
import nltk
import re
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Lambda
from keras.models import Model


lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
stopwords.update(["et", "al", "mr"])
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1, lowercase=False)


def remove(text):
    remove_chars = '[0-9’"#$%&\'()*+,-./:;<=>@，。★、…【】《》“”‘’[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if word not in stopwords:
            if lemmatizer.lemmatize(word, tag) == "co":
                lemmatized_sentence.append("carbon")
            if lemmatizer.lemmatize(word, tag) != "c":
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def preprocess(events, dataset):
    ques = '[?？]+'
    excl = '[!！]+'
    for event in events:
        paragragh = lemmatize_sentence(remove(event))
        paragragh_1 = re.sub(ques, 'QUESTION MARK', paragragh)
        paragragh_final = re.sub(excl, 'EXCLAMATION MARK', paragragh_1)
        dataset.append(paragragh_final)


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
print("preprocess done")


train_transformed = tfidf_vectorizer.fit_transform(X_train)
x_train = train_transformed.toarray()
dev_transformed = tfidf_vectorizer.transform(X_dev)
x_dev = dev_transformed.toarray()
test_transformed = tfidf_vectorizer.transform(X_test)
x_test = test_transformed.toarray()
features = (tfidf_vectorizer.get_feature_names())

# Top rank bigrams for Error analysis
# sums = train_transformed.sum(axis = 0)
# data1 = []
# for col, term in enumerate(features):
#     data1.append( (term, sums[0, col] ))
# ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
# words = (ranking.sort_values('rank', ascending = False))
# print ("\n\nTrain_Words : \n", words.head(50))
#
# sums = dev_transformed.sum(axis = 0)
# data1 = []
# for col, term in enumerate(features):
#     data1.append( (term, sums[0, col] ))
# ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
# words = (ranking.sort_values('rank', ascending = False))
# print ("\n\nDev_Words : \n", words.head(50))

vocab_size_1 = x_train.shape[1]
print("Vocab size =", vocab_size_1)

# Input transformed for Bigram TF-IDF bidirectional LSTM Model
# x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
# x_dev = x_dev.reshape((x_dev.shape[0], 1, x_dev.shape[1]))
# bert_x_test = bert_x_test.reshape((bert_x_test.shape[0], 1, bert_x_test.shape[1]))

# Find the Optimal parameter C for linear regression
# accuracies_LR = []
# c_to_test = [0.001,0.005,0.010,0.050,0.100,0.500,1.000,5.000,10.00,50.00,100.0,500.0,1000]
# for c_test in c_to_test:
#     clf_LR = LogisticRegression(C=c_test)
#     clf_LR.fit(x_train, y_train)
#     predictions = clf_LR.predict(x_dev)
#     print("LogisticRegression(C= " + str(c_test) + ")")
#     f1 = f1_score(y_dev, predictions)
#     accuracies_LR.append(f1)
#     print("f1 score", f1)
#     print(classification_report(y_dev, predictions))
# best_C = c_to_test[accuracies_LR.index(max(accuracies_LR))]
# print("Optimal parameter C is " + str(best_C))

# Bigram TF-IDF Linear Regression model
# clf_LR = LogisticRegression(C=50)
# clf_LR.fit(x_train, y_train)
# pred = clf_LR.predict(x_dev)
# print(clf_LR)
# print(classification_report(y_dev, pred))
#
# test_predictions = clf_LR.predict(x_test)


# Test predict
# for i in range(0, len(test_predictions)):
#     result_test.update({'test-%d' % i: {'label': int(test_predictions[i])}})
#
# with open("test-output.json", "w") as ftow:
#     json.dump(result_test, ftow)

print("data is ready")
# Bigram TF-IDF forward neural network model
model1 = Sequential(name="Bigram TF-IDF forward neural network model")
model1.add(layers.Dense(10, input_dim=vocab_size_1, activation='relu'))
model1.add(layers.Dropout(0.2))
model1.add(layers.Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()
model1.fit(x_train, y_train, epochs=2, verbose=True, validation_data=(x_dev, y_dev), batch_size=10)
predictions = model1.predict_classes(x_dev, verbose=True, batch_size=10)


# Bigram TF-IDF bidirectional LSTM Model
# input_layer = Input((None,vocab_size_1))
# model = Dense(256, activation='relu')(input_layer)
# model = Bidirectional(LSTM(units=512, return_sequences=True))(model)
# model = Bidirectional(LSTM(64, return_sequences=False))(model)
# pred = Dense(1, activation='sigmoid')(model)
# model = Model(inputs=input_layer, outputs=pred)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# model.fit(x_train, y_train , epochs=3, verbose=True, validation_data=(x_dev, y_dev), batch_size=10)
# predictions = model.predict(x_dev, verbose=True, batch_size=10)

# Test data prediction output
# for i in range(0, len(predictions)):
#     wordlist = nltk.word_tokenize(X_test[i])
#     if "climate" not in wordlist:
#         result_test.update({'test-%d' % i: {'label': 0}})
#     else:
#         result_test.update({'test-%d' % i: {'label': int(predictions[i])}})
# with open("test-output.json", "w") as ftow:
#     json.dump(result_test, ftow)

# Dev data prediction output
for i in range(0, len(predictions)):
    result_dev.update({'dev-%d' % i: {'label': int(predictions[i])}})
    print(i, y_dev[i], predictions[i])
with open("dev_result.json", "w") as ftow:
    json.dump(result_dev, ftow)

print("Done")
