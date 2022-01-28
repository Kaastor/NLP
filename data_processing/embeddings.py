# Embeddings - data representation
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s, l in train_data:  # sentence, label
    training_sentences.append(str(s.numpy()))
    training_labels.append(str(l.numpy()))

for s, l in test_data:  # sentence, label
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(str(l.numpy()))

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# Neural Network for
model = tf.keras.Sequential([
    # shape (# of words in sentence x emb_dim)
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # because embeddings are 2D. Flatten - reshape a tensor into 2D
    # or GlobalAveragePooling1D - average all the values according to the last axis (faster, less parameters)
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# FIXME: throws error?
num_epochs = 10
with tf.device('/gpu:0'):
    model.fit(training_padded, training_labels_final, epochs=num_epochs,
              validation_data=(test_padded, testing_labels_final))
