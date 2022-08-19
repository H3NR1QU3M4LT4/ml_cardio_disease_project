import tensorflow as tf
import tensorflow_addons as tfa
import datetime

def define_model(feature_columns):
    model = tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tfa.metrics.F1Score(num_classes=1,
                                               average='macro',
                                               threshold=0.5)])

    return model


def train(model, train_ds, test_ds):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=1000,
                        use_multiprocessing=True,
                        callbacks=[tensorboard_callback])

    model.evaluate(test_ds)

    return history
