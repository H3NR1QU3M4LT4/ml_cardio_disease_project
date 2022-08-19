import pandas as pd
from absl import logging


def load_data(csv_path):
    logging.info('Loading CSV file')

    dataset = pd.read_csv(csv_path, sep=',')
    dataframe = pd.DataFrame(dataset)

    logging.info(f'CSV loaded as a dataframe:\n {dataframe}')

    return dataframe
