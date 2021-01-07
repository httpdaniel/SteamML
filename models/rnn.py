# import packages
seed = 13
import os
os.environ['PYTHONHASHSEED'] = str(seed)
import random
random.seed(seed)
from keras.layers import *
from tensorflow.keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed)

# read in reviews
reviews = pd.read_csv('../data/transformed_reviews.csv')
reviews.fillna(value='', inplace=True)

# tokenize the review text
max_words = 26000
max_len = 400
tokenizer = Tokenizer(split=' ', num_words=max_words)
tokenizer.fit_on_texts(reviews['Review'].values)
X = tokenizer.texts_to_sequences(reviews['Review'].values)
X = pad_sequences(X, max_len)

# add length column
X = np.column_stack((X, reviews['Length'].values))

# encode labels
Y = pd.get_dummies(reviews['Recommended']).values

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

embedding_size = 100
lstm_out = 128

# build model
model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=max_len+1))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.summary()

# fit the model
batch_size = 64
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1, batch_size=batch_size)

# plot loss and accuracy over epoch
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# evaluate predictions on training data
preds = model.predict(X_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1, y_pred))

# evaluate predictions on test data
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1, y_pred))
