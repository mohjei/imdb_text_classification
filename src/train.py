
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from timeutils import Stopwatch
from src.config import MlConf
from src.utils.path_helper import logger, MODEL_DIR, ASSET_DIR
from src.utils.ml_helper import clf_metrics


class RocAucEvaluation(tf.keras.callbacks.Callback):

    def __init__(self, validation_data=(), interval=1):
        super(tf.keras.callbacks.Callback, self).__init__()

        self.interval = interval
        self.x_val, self.y_val = validation_data


class CnnTrain:

    def __init__(self):

        self.max_features = MlConf.max_features
        self.max_len = MlConf.max_len
        self.filters = MlConf.filters
        self.kernel_size_1 = MlConf.kernel_size_1
        self.kernel_size_2 = MlConf.kernel_size_2
        self.kernel_size_3 = MlConf.kernel_size_3
        self.kernel_size_4 = MlConf.kernel_size_4
        self.dropout_rate = MlConf.dropout_rate
        self.hidden_dense = MlConf.hidden_dense
        self.out_dense = MlConf.out_dense
        self.hidden_activation = MlConf.hidden_activation
        self.out_activation = MlConf.out_activation
        self.learning_rate = MlConf.learning_rate
        self.batch_size = MlConf.batch_size
        self.epochs = MlConf.epochs

    @staticmethod
    def get_class_weights(y):

        y_integers = np.argmax(y, axis=1)

        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)

        return dict(enumerate(class_weights))
    
    
    def cnn_model(self):

        inputs = tf.keras.Input(shape=(None,))

        x = tf.keras.layers.Embedding(input_dim=self.max_features, output_dim=self.max_len, trainable=True)(inputs)

        x = tf.keras.layers.SpatialDropout1D(self.dropout_rate)(x)

        c1 = tf.keras.layers.Conv1D(self.filters, self.kernel_size_1, activation=self.hidden_activation)(x)

        max_pool2 = tf.keras.layers.GlobalMaxPooling1D()(c1)

        c2 = tf.keras.layers.Conv1D(self.filters, self.kernel_size_2, activation=self.hidden_activation)(x)

        max_pool3 = tf.keras.layers.GlobalMaxPooling1D()(c2)

        c3 = tf.keras.layers.Conv1D(self.filters, self.kernel_size_3, activation=self.hidden_activation)(x)

        max_pool4 = tf.keras.layers.GlobalMaxPooling1D()(c3)

        c4 = tf.keras.layers.Conv1D(self.filters, self.kernel_size_4, activation=self.hidden_activation)(x)

        max_pool5 = tf.keras.layers.GlobalMaxPooling1D()(c4)

        conc = tf.keras.layers.concatenate([max_pool2, max_pool3, max_pool4, max_pool5])

        conc = tf.keras.layers.Dense(self.hidden_dense)(conc)

        output = tf.keras.layers.Dense(self.out_dense, activation=self.out_activation, name='output')(conc)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.summary()

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                        optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                        metrics=tf.keras.metrics.AUC(name='auc'))

        return model

    def cnn_fit(self, x_train, x_test, y_train, y_test):

        model = self.cnn_model()

        roc_auc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)

        sw = Stopwatch(start=True)

        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[roc_auc])

        logger.info(f'Training time: {sw.elapsed.human_str()}')

        tf.keras.models.save_model(model, MODEL_DIR)

        y_pred = model.predict(x_test)

        metric_dict = clf_metrics(y_test, y_pred)

        logger.info(f'Model evaluation using test set: ')
        for k, v in metric_dict.items():
            logger.info(f'{k} score: {v}')

        return model
