import tensorflow as tf


def attributes_encoding_to_tensorflow(dataframe):
    feature_columns = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # bucketized cols
    age = tf.feature_column.numeric_column('age')
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    dataframe['thal'] = dataframe['thal'].apply(str)
    thal = tf.feature_column.categorical_column_with_vocabulary_list(
        'thal', ['3', '6', '7'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    dataframe['sex'] = dataframe['sex'].apply(str)
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        'sex', ['0', '1'])
    sex_one_hot = tf.feature_column.indicator_column(sex)
    feature_columns.append(sex_one_hot)

    dataframe['cp'] = dataframe['cp'].apply(str)
    cp = tf.feature_column.categorical_column_with_vocabulary_list(
        'cp', ['0', '1', '2', '3'])
    cp_one_hot = tf.feature_column.indicator_column(cp)
    feature_columns.append(cp_one_hot)

    dataframe['slope'] = dataframe['slope'].apply(str)
    slope = tf.feature_column.categorical_column_with_vocabulary_list(
        'slope', ['0', '1', '2'])
    slope_one_hot = tf.feature_column.indicator_column(slope)
    feature_columns.append(slope_one_hot)

    # embedding cols
    thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    # crossed cols
    age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
    feature_columns.append(age_thal_crossed)

    cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
    cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
    feature_columns.append(cp_slope_crossed)

    return feature_columns
