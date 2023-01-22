# Import libraries
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load the IMDB review dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

# Preprocess the data
x_train = pad_sequences(x_train, maxlen=80)
x_test = pad_sequences(x_test, maxlen=80)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Build the model
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))

# Print the results
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Plot the training and validation accuracy and loss over the epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot the training and validation loss over the epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Get the index of a random positive review
good_review_index = np.random.randint(0, x_test.shape[0] - 1)
while y_test[good_review_index][0] != 0:
    good_review_index = np.random.randint(0, x_test.shape[0] - 1)
    
# Get the word index
word_index = imdb.get_word_index()

# Create a reverse index mapping integers to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode the good review
good_review = " ".join([reverse_word_index.get(i - 1, '?') for i in x_test[good_review_index]])
print("Good review:")
print(good_review)

# Decode the bad review
bad_review = " ".join([reverse_word_index.get(i - 1, '?') for i in x_test[bad_review_index]])
print("Bad review:")
print(bad_review)

# Get the index of a random negative review
bad_review_index = np.random.randint(0, x_test.shape[0] - 1)
while y_test[bad_review_index][0] != 1:
    bad_review_index = np.random.randint(0, x_test.shape[0] - 1)

# Print the good and bad reviews
good_review = " ".join([imdb.get_word_index()[idx] for idx in x_test[good_review_index]])
print("Good review:")
print(good_review)

bad_review = " ".join([imdb.get_word_index()[idx] for idx in x_test[bad_review_index]])
print("Bad review:")
print(bad_review)
