from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn.metrics import classification_report
import numpy as np


class Bi_LSTM(object):
    def __init__(self, maxlen, n_words, n_tags):
        input_ = Input(shape=(maxlen,))
        model = Embedding(input_dim=n_words, output_dim=maxlen, input_length=maxlen)(input_)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

        self.model = Model(input_, out)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, X_train, y_train, epoch=1, batch_size=32):
        self.model.fit(X_train, np.array(y_train),
                       batch_size=batch_size, epochs=epoch,
                       validation_split=0.2, verbose=1)
        self.model.summary()

    def evaluate(self, X_test, y_test, tags):
        y_pred = self.model.predict(np.array([X_test]))
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = y_pred.ravel()
        return (classification_report(y_pred=y_pred, y_true=y_test,
                target_names=tags))
