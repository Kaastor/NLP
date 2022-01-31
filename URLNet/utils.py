import csv
import itertools

from matplotlib import pyplot as plt
from nltk import RegexpTokenizer
from nltk.probability import FreqDist
from typing import List

OOV_TOKEN = '<OOV>'
OOV_TOKEN_ID = 1
PAD_TOKEN = '<PAD>'
PAD_TOKEN_ID = 0


def load_data(file_path):
    x = []
    y = []
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        # next(reader)
        for row in reader:
            y.append(int(row[0]))
            x.append(row[1])

    return x, y


def tokenize_by_words(urls, encoding_max_len=200):
    """
    Word level tokenizing. Returns encoded, padded tokens

    In word_x sometimes divide in words? E.x NedMoney -> Ned, Money
    Tensorflow Tokenizer does not provide an option for custom tokenizing.
    For that reason we need to build custom word_index for specified corpus.

    :param encoding_max_len: length of encoding, not used places will be padded
    :param urls: list of strings (URLs)
    """
    tokenizer = RegexpTokenizer('[\w-]+|[^\w\s]')
    tokenized_text = tokenizer.tokenize_sents(urls)
    words_freq = most_frequent_tokens(tokenized_text, None)

    vocab = build_token_index(words_freq)

    return tokenized_text, encode_sentences(tokenized_text, vocab, encoding_max_len), vocab


def tokenize_by_chars(urls, encoding_max_len=200, min_char_frequency=100):
    """
    Char level tokenizing. Returns encoded, padded tokens.

    :param min_char_frequency: leave only chars which happen most frequently
    :param encoding_max_len: length of encoding, not used places will be padded
    :param urls: list of strings (URLs)
    """
    tokenized_text = [[char for char in url] for url in urls]
    f_dist = FreqDist(itertools.chain.from_iterable(tokenized_text))
    remove_less_frequent_tokens(f_dist, min_char_frequency)
    chars_freq = most_frequent_tokens(tokenized_text, None)

    vocab = build_token_index(chars_freq)
    return encode_sentences(tokenized_text, vocab, encoding_max_len), vocab


def tokenize_by_words_in_chars(word_sequences, vocab, encoding_max_len=200):
    """
    Words in Chars level tokenizing. Returns encoded, padded tokens like [[h, t, t, p], [..]]
    TODO: should I add start and end token '<' & '>'? E.x: ['<', 'h', 't', 't', 'p', '>']
    :param encoding_max_len: length of encoding, not used places will be padded
    :param vocab: vocabulary {char: id}
    :param word_sequences: list of lists of word tokens: ['http', ':', '/', '/', 'daappconnections', '.', 'com', ...]
    """
    tokenized_by_chars = [[[char for char in word]
                           for word in words]
                          for words in word_sequences]

    return [encode_sentences(sentence, vocab, encoding_max_len) for sentence in tokenized_by_chars]


def encode_sentences(tokenized_sentences: List[List[str]], vocab, encoding_max_len=200):
    """
    :param encoding_max_len: length of the encoding, not filled elements will be padded
    :param tokenized_sentences: list of lists of string tokens
    :param vocab: dict: {'token': id}
    """
    encoded_sentences = []
    for sentence in tokenized_sentences:
        encoded_sentences.append(tokenize_sentence(sentence, vocab, encoding_max_len))
    return encoded_sentences


def tokenize_sentence(tokenized_sentence: List[str], vocab, encoding_max_len=200):
    """
    :param encoding_max_len: length of the encoding, not filled elements will be padded
    :param tokenized_sentence: list of string tokens
    :param vocab: dict: {'token': id}
    """
    encoded = []
    for token in tokenized_sentence[:encoding_max_len]:
        encoding = vocab[token]
        encoded.append(encoding if encoding else vocab[OOV_TOKEN])
    i = len(encoded)
    while i < encoding_max_len:
        encoded.append(PAD_TOKEN_ID)
        i += 1

    return encoded


def tokenized_to_string(tokenized_sequence, word_index):
    """
    Changes numerical tokenized sequence back to string based on dictionary
    :param tokenized_sequence: List of numbers
    :param word_index: dictionary ["string" -> token]
    :return: List of strings
    """
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return [reverse_word_index[token] for token in tokenized_sequence]


def most_frequent_tokens(tokenized_string_sequences, most_frequent_number=None):
    """
    :param tokenized_string_sequences: List of Lists of strings
    :param most_frequent_number
    :return: list of tuples: ('string', occurrences_number)
    """
    f_dist = FreqDist(itertools.chain.from_iterable(tokenized_string_sequences))
    return f_dist.most_common(most_frequent_number)


def build_token_index(tokens_freq):
    """
    Count starts from 2, because 0 is reserved for padding and 1 for <OOV>
    :param tokens_freq: dict: {'string': occurrences_number}
    """
    count = 2
    token_index = {PAD_TOKEN: PAD_TOKEN_ID, OOV_TOKEN: OOV_TOKEN_ID}
    for (token, freq) in tokens_freq:
        token_index[token] = count
        count += 1
    return token_index


def remove_less_frequent_tokens(f_dist: FreqDist, min_char_frequency):
    to_remove_keys = []
    for item in f_dist.items():
        if item[1] < min_char_frequency:
            to_remove_keys.append(item[0])
    for key in to_remove_keys: f_dist.pop(key)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.savefig("./plots/" + string + ".png")
    plt.show()
