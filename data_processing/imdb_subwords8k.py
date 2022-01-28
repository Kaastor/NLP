# https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder
# Case sensitive!
# Neural networks can efficiently work with only ~30 thousands tokens.
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder  # SubwordTextEncoder trained on imdb dataset. The most frequent sub-words
print(tokenizer.subwords)

sample = 'tensorflow, from basics to master in a week'

tokenized_string = tokenizer.encode(sample)
print(tokenized_string)

original_string = tokenizer.decode(tokenized_string)
print(original_string)

for ts in tokenized_string:
    print('{} ---> {}'.format(ts, tokenizer.decode([ts])))


# Data

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

# Neural Network
embeddings_dim = 64
model = tf.keras.Sequential([
    # shape (# of words in sentence x emb_dim)
    tf.keras.layers.Embedding(tokenizer.vocab_size, embeddings_dim),
    # because embeddings are 2D. Flatten - reshape a tensor into 2D
    # or GlobalAveragePooling1D - average all the values according to the last axis (faster, less parameters)
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# binary_crossentropy because two classes?
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(train_dataset,  # FIXME for 'train_data' not working
                    epochs=num_epochs,
                    validation_data=test_dataset,
                    verbose=1)

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


