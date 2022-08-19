import tensorflow as tf
from sklearn.model_selection import train_test_split


def define_target(dataframe, column_name):
    dataframe = dataframe.rename(columns={column_name: 'target'})
    return dataframe


def create_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) \
        .shuffle(buffer_size=len(dataframe)) \
        .batch(batch_size)


def create_train_test_datasets(dataframe, random_seed):
    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=random_seed)
    train_ds = create_dataset(train_df)
    test_ds = create_dataset(test_df)
    return train_ds, test_ds, train_df, test_df
