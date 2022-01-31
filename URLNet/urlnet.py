from URLNet.URLNetModel import URLNetModel
from utils import *

data_dir = "./data/urls.txt"
X, y = load_data(data_dir)

# Word level - tokenized by words
params_list = [{
    "chars_embedding_dim": 32,
    "encoding_max_len": 200,
    "batch_size": 32,
    "epochs": 5
}]

for params in params_list:
    # Word-level
    # word_sequences, word_sequences_encoded, word_index = tokenize_by_words(X)

    # Char level - tokenized by chars
    char_sequences_encoded, char_vocab = tokenize_by_chars(X, params["encoding_max_len"])

    # Words in chars - tokenized by chars in words
    # words_in_char_sequences_encoded = tokenize_by_words_in_chars(word_sequences, char_vocab)

    experiment = "Experiment with Char-level & {} emb_dim".format(params['chars_embedding_dim'])
    X = char_sequences_encoded
    model = URLNetModel(params, len(char_vocab))
    (experimentID, runID) = model.mlflow_run(X, y, run_name="Phishing URL Classification Model, URLNet Inspired")
    print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))

