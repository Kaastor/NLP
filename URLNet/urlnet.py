import numpy_ml
from keras import Sequential
from sklearn.model_selection import train_test_split

from utils import *
from tensorflow.keras.layers import *

'''Implantation of URLNet.'''
data_dir = "data/urls.txt"

X, y = load_data(data_dir)

''' 
Data Pre-Processing
Prepare data to be used in Embeddings Layers

'''
# Word level - tokenized by words
chars_embedding_dim = 32
encoding_max_len = 200
batch_size = 32

word_sequences, word_sequences_encoded, word_index = tokenize_by_words(X)
# print(tokenized_to_string(word_level_sequences[1], word_index))

# Char level - tokenized by chars
char_sequences_encoded, char_vocab = tokenize_by_chars(X, encoding_max_len)
# print(tokenized_to_string(char_level_sequences[1], char_index))

# Words in chars - tokenized by chars in words
words_in_char_sequences_encoded = tokenize_by_words_in_chars(word_sequences, char_vocab)
# print([tokenized_to_string(word, char_index) for word in words_in_char_sequences_encoded[0]])

X = char_sequences_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''1. Char-level CNN'''
model = Sequential([
    # shape (# of chars in sentence x emb_dim)
    # fixme: maybe chars_embedding_dim could be shorter
    # because embeddings are 2D. Flatten - reshape a tensor into 2D
    # or GlobalAveragePooling1D - average all the values according to the last axis (faster, less parameters)
    Embedding(len(char_vocab),
              chars_embedding_dim,
              name='embedding',
              input_length=encoding_max_len),
    Dropout(0.5),
    # CNN
    Conv1D(256,
           3,
           name='conv_1',
           # valid = no padding (trim leftovers)
           padding='valid',
           activation='relu',
           input_shape=(batch_size, encoding_max_len, chars_embedding_dim)),
    MaxPooling1D(),
    Conv1D(256,
           4,
           name='conv_2',
           padding='valid',
           activation='relu'),
    MaxPooling1D(),
    Conv1D(256,
           5,
           name='conv_3',
           padding='valid',
           activation='relu'),
    MaxPooling1D(),
    Conv1D(256,
           6,
           name='conv_4',
           padding='valid',
           activation='relu'),
    MaxPooling1D(),
    # FC
    Flatten(),
    # Dense(250, activation='relu'),
    # Dropout(0.5),
    # Dense(1, activation='sigmoid')
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=8,
                    validation_data=(X_test, y_test))

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
