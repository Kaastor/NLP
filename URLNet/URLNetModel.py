import mlflow
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Embedding
from sklearn.model_selection import train_test_split

from URLNet.utils import plot_graphs

'''Char-level CNN'''


class URLNetModel:

    def __init__(self, params=None, char_vocab_len=None):
        if params is None:
            params = {}
        self._params = params
        self.model = Sequential([
            Embedding(char_vocab_len,
                      params["chars_embedding_dim"],
                      name='embedding',
                      input_length=params["encoding_max_len"]),
            Dropout(0.5),
            # CNN
            Conv1D(256,
                   3,
                   name='conv_1',
                   # valid = no padding (trim leftovers)
                   padding='valid',
                   activation='relu',
                   input_shape=(params["batch_size"], params["encoding_max_len"], params["chars_embedding_dim"])),
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
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    @property
    def params(self):
        """
        Getter for model parameters
        returns: Dictionary of model parameters
        """
        return self._params

    def mlflow_run(self, X, y, run_name="URLNet Model", verbose=False):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param verbose:
        :param run_name: Name of the experiment run as logged by MLflow
        :return: Tuple of MLflow experimentID, runID
        """
        with mlflow.start_run(run_name=run_name) as run:
            # get experimentalID and run_id
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=['Accuracy', 'Precision', 'Recall'])

            mlflow.keras.autolog()

            history = self.model.fit(X_train, y_train,
                                     batch_size=self.params["batch_size"],
                                     epochs=self.params["epochs"],
                                     validation_data=(X_test, y_test))

            # Log metrics
            mlflow.log_metric("Accuracy", history.history['Accuracy'][-1])
            mlflow.log_metric("Precision", history.history['precision'][-1])
            mlflow.log_metric("Recall", history.history['recall'][-1])

            plot_graphs(history, "Accuracy")
            mlflow.log_artifact("./plots/Accuracy.png")
            plot_graphs(history, "recall")
            mlflow.log_artifact("./plots/recall.png")
            plot_graphs(history, "loss")
            mlflow.log_artifact("./plots/loss.png")

            mlflow.log_params(self.params)

            return experiment_id, run_id
