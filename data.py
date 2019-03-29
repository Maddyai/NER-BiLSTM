from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import numpy as np


def pre_process(data_path):
    data = pd.read_csv(data_path, sep=r'\n', header=None, engine='python')
    data.columns = ['text']

    b = pd.DataFrame(data.text.str.split(' ').tolist()).stack()
    train = b.reset_index()

    train.columns = ["sentence_idx", "words_in_sent", "text"]

    train[['word', 'pos_tag', 'tag']] = train['text'].str.split('|',
                                                                expand=True)
    train = train.drop(['words_in_sent', 'text', 'pos_tag'], axis=1)
    return train  # , test_df


def prepare_config(dataset):
    words = list(set(dataset["word"].values))
    words.append("ENDPAD")
    n_words = len(words)

    tags = list(set(dataset["tag"].values))
    n_tags = len(tags)

    return n_words, words, n_tags, tags


def post_process(sentences, n_words, words, n_tags, tags):
    maxlen = max([len(s) for s in sentences])

    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    X = pad_sequences(maxlen=maxlen, sequences=X, padding="post",
                      value=n_words - 1)
    y = pad_sequences(maxlen=maxlen, sequences=y, padding="post",
                      value=tag2idx["O"])

    y = [to_categorical(i, num_classes=n_tags) for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    y_test = np.array(y_test)
    y_test = np.argmax(y_test, axis=-1)
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test, maxlen


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s

        except:
            return None
