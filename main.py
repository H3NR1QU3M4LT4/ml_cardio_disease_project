# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from absl import app, flags
from prepare_data.load_data import load_data
from prepare_data.get_information_from_data import get_information_from_data
from prepare_data.attributes_encoding_to_tensorflow import attributes_encoding_to_tensorflow
from prepare_data.dataset_for_trainning import create_train_test_datasets, define_target
from modeling.model import define_model, train
from modeling.confusion_matrix import get_confusion_matrix

FLAGS = flags.FLAGS

flags.DEFINE_string('PATH_CSV', 'data/', 'CSV file path')
flags.DEFINE_string('SAVE_REPORT_PATH', 'data/', 'Folder where the pandas profiling should be stored')


def main(_):

    dataframe = load_data(FLAGS.PATH_CSV)
    get_information_from_data(dataframe, FLAGS.PATH_CSV)

    dataframe = define_target(dataframe, 'condition')
    feature_columns = attributes_encoding_to_tensorflow(dataframe)
    train_ds, test_ds, train_df, test_df = create_train_test_datasets(dataframe, 42)

    model = define_model(feature_columns)

    history = train(model, train_ds, test_ds)

    get_confusion_matrix(model, test_ds, test_df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(main)
