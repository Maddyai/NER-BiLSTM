from data import pre_process, prepare_config, post_process, SentenceGetter
from model import Bi_LSTM


if __name__ == "__main__":
    data_path = "data/aij-wikiner-en-wp2"

    dataset = pre_process(data_path)
    n_words, words, n_tags, tags = prepare_config(dataset)
    getter = SentenceGetter(dataset)

    X_train, X_test, y_train, y_test, maxlen = post_process(
        getter.sentences, n_words, words, n_tags, tags)

    lstm_model = Bi_LSTM(maxlen, n_words, n_tags)
    lstm_model.train(X_train, y_train, epoch=1)

    result = lstm_model.model.evaluate(X_test, y_test, tags)

    print("[INFO] Final accuracy :{} of the model".format(result))
