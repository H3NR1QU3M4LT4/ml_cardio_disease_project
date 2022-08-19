import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def get_confusion_matrix(model, test_ds, test_df):
    predictions = model.predict(test_ds)
    bin_predictions = tf.round(predictions).numpy().flatten()
    true_values = test_df['target'].to_numpy().astype(float)

    print(classification_report(true_values, bin_predictions))
