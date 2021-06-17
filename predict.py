
import numpy as np
import pickle
import pandas as pd
from timeutils import Stopwatch
from src.utils.path_helper import ASSET_DIR, DATA_DIR, logger, MODEL_DIR
from src.utils.ml_helper import text_to_seq, pad_seq
from src.utils.clean import clean_text
import tensorflow as tf


def load_tokeizer(filename=ASSET_DIR / 'token.pkl'):

    try:
        with filename.open('rb') as handle:
            return pickle.load(handle)
    except Exception:
        raise ValueError('Failed to load the desired file')


def data_prep(x):

    token = load_tokeizer(filename=ASSET_DIR / 'token.pkl')

    x = clean_text(x)

    x = text_to_seq(token, x)

    x = pad_seq(x)

    return x


def load_model():
    return tf.keras.models.load_model(MODEL_DIR)


def predict(model, x):

    y = model.predict(x)
    
    y = np.argmax(y, axis=1)

    return np.where(y >= 0.5, 1, 0)


if __name__ == '__main__':

    sw = Stopwatch(start=True)

    x = "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time."
    
    x = data_prep(x)

    model = load_model()

    labels = predict(model, x)

    logger.info(f'labels: {labels}')

    logger.info(f'Total elapsed time: {sw.elapsed.human_str()}')
