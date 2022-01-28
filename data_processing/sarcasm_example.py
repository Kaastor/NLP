# Sarcasm detection
# https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home
# is_sarcastic: 1
# headline: headline of article
# article_link: link to article for supplementary data
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


dataset_file = "/home/przemek/Deep Learning/road-to-machine-learning/NLP/data_processing/data/sarcasm.json"
# by reducing vocab size, we reduce words being discovered by the tokenizer.
# This has no impact on the corpus! Corpus (dictionary) will discover all the words. This
# has impact how many of words will be discovered in `texts_to_sequences` process.
# Only the most common `num_words-1` words will be kept.
vocab_size = 2700  # 2700
embeddings_dim = 16
# This will reduce overall sentence length (thus reducing padding). For example most words have length of 5, but few
# of them - 100. Padding most of them to fill the 100 would be to much 'noisy'.
max_length = 25  # 25
trunc_type = 'post'
paddings_type = 'post'
oov_tok = "<OOV>"
training_size = 20000  # (7k left for test)

sentences = []
labels = []
urls = []

for item in open(dataset_file, 'r'):
    sentences.append(json.loads(item)['headline'])
    labels.append(json.loads(item)['is_sarcastic'])
    # urls.append(json.loads(item)['article_link'])

train_sentences = sentences[0:training_size]
test_sentences = sentences[training_size:]
train_labels = labels[0:training_size]
test_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=paddings_type, maxlen=max_length, truncating=trunc_type).tolist()
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding=paddings_type, maxlen=max_length, truncating=trunc_type).tolist()

# Neural Network for
model = tf.keras.Sequential([
    # shape (# of words in sentence x emb_dim)
    tf.keras.layers.Embedding(vocab_size, embeddings_dim, input_length=max_length),
    # because embeddings are 2D. Flatten - reshape a tensor into 2D
    # or GlobalAveragePooling1D - average all the values according to the last axis (faster, less parameters)
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# binary_crossentropy because two classes?
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

num_epochs = 30
history = model.fit(train_padded, train_labels,
                    epochs=num_epochs,
                    validation_data=(test_padded, test_labels),
                    verbose=1)

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
# loss - confidence in a prediction. If loss is increasing in value (decreasing in confidence) then we need to tweak
# hyper-parameters for over 90% accuracy without loss increasing sharply
