
import tensorflow as tf
import numpy as np
from sklearn.metrics import *
from src.config import MlConf
from src.utils.path_helper import MODEL_DIR


def tokenizer(text, max_features=MlConf.max_features):

    tk = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, oov_token='UNK')

    tk.fit_on_texts(text)

    return tk


def text_to_seq(tk, data):
    return tk.texts_to_sequences(data)


def pad_seq(seq, maxlen=MlConf.max_len):
    x = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen)
    return x


def clf_metrics(y_true, y_pred):

    y_label = np.where(y_pred >= 0.5, 1, 0)

    metric_dict = dict()

    acc = np.round(accuracy_score(y_true, y_label), 2)

    metric_dict.update({'accuracy': acc})

    auc = np.round(roc_auc_score(y_true, y_pred), 2)

    metric_dict.update({'AUC': auc})

    f1 = np.round(f1_score(y_true, y_label), 2)

    metric_dict.update({'F1': f1})

    print(classification_report(y_true, y_label))

    return metric_dict


def prediction(x):
    model = tf.keras.models.load_model(MODEL_DIR)

    return model.predict(x)
