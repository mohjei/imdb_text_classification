
from src.etl import DataPrep
from src.train import CnnTrain
from timeutils import Stopwatch
from src.utils.path_helper import logger


if __name__ == '__main__':

    sw = Stopwatch(start=True)

    data_prep = DataPrep(fname='reviews.csv')

    x_train, x_test, y_train, y_test = data_prep.prep_data()

    cnn_train = CnnTrain()

    cnn_train.cnn_fit(x_train, x_test, y_train, y_test)

    logger.info(f'Total elapsed time: {sw.elapsed.human_str()}')
