from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog',
    'Do you think my dog is amazing?'
]

# 1. Create a dictionary (corpus)
tokenizer = Tokenizer(num_words=30, oov_token='<OOV>')  # num_words - number of distinct words in text
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# 2. Use Corpus to create sequences (words) based on values from tokenizer
# If we use sentences with words not included in Tokenizer - they will be omit
# We can add oov_token in Tokenizer - it will replace not known words
sequences = tokenizer.texts_to_sequences(sentences)

# Create uniform size of the sequence. Shorter sentences will be filled with zeros.
padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')  # padding='post' - add padding after the seq
# 3. Test

test_data = [
    'i really love my dog',
    'my dog loves my manatee lol'
]
test_seq = tokenizer.texts_to_sequences(test_data)
padded_test = pad_sequences(test_seq, padding='post')

print(word_index)
print(padded_test)
