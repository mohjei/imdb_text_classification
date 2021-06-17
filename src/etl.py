
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from timeutils import Stopwatch
from src.config import EtlConf
from src.utils.path_helper import ASSET_DIR, DATA_DIR, logger
from src.utils.ml_helper import tokenizer, text_to_seq, pad_seq
from src.utils.clean import clean_text


class DataPrep:

    def __init__(self, fname):

        self.test_size = EtlConf.test_size
        self.val_size = EtlConf.val_size
        self.fname = fname

    def get_data(self):

        return pd.read_csv(DATA_DIR / self.fname)

    @staticmethod
    def dump(obj, filename):

        try:
            with filename.open('wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            raise ValueError('Failed to dump the desired file')

    def prep_data(self):

        sw = Stopwatch(start=True)

        logger.info(f'Reading dataset')

        df = self.get_data()

        logger.info(f'Cleaning reviews')

        cleaned_reviews = df['review'].apply(clean_text)

        # df = df.assign(label=lambda x: np.where(x.sentiment == 'positive', 1, 0))

        x_train, x_test, y_train, y_test = \
            train_test_split(cleaned_reviews,
                             df[['label']],
                             test_size=self.test_size,
                             stratify=df[['label']],
                             shuffle=True,
                             random_state=0)

        logger.info(f'Tokenizing reviews')

        # Tokenize
        token = tokenizer(x_train)

        logger.info(f'Converting texts into sequences')
        # texts_to_sequences
        x_train = text_to_seq(token, x_train)
        x_test = text_to_seq(token, x_test)

        logger.info(f'Padding sequences')
        # padding
        x_train = pad_seq(x_train)
        x_test = pad_seq(x_test)

        self.dump(token, ASSET_DIR / 'token.pkl')

        logger.info(f'ETL time: {sw.elapsed.human_str()}')

        return x_train, x_test, y_train, y_test
