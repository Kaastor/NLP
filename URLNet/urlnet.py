from utils import *

'''Implantation of URLNet.'''
data_dir = "data/test_urls.csv"

urls, labels = load_data(data_dir)
''' 
Data Pre-Processing
Prepare data to be used in Embeddings Layers

'''
# Word level - tokenized by words

word_sequences, word_sequences_encoded, word_index = tokenize_by_words(urls)
# print(tokenized_to_string(word_level_sequences[1], word_index))

# Char level - tokenized by chars
char_sequences_encoded, char_index = tokenize_by_chars(urls)
# print(tokenized_to_string(char_level_sequences[1], char_index))

# Words in chars - tokenized by chars in words
words_in_char_sequences_encoded = tokenize_by_words_in_chars(word_sequences, char_index)
# print([tokenized_to_string(word, char_index) for word in words_in_char_sequences_encoded[0]])

'''1. Char-level CNN'''
