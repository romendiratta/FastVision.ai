import tensorflow as tf
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk


def get_dataset(epochs, batch_size, data_dir, channel_name):

    def _dataset_parser(data, label):
        file_path = os.path.join(data_dir, data.numpy().decode("utf-8") + ".mhd")
        ct_scan = sitk.ReadImage(file_path)
        ct_scan = sitk.GetArrayFromImage(ct_scan)
        return ct_scan, label

    def _expand_dims(data, label):
        return tf.expand_dims(data, axis=3), label

    metadata = pd.read_csv(os.path.join(data_dir, channel_name + ".csv"))

    dataset = tf.data.Dataset.from_tensor_slices((metadata["seriesuid"].values,
                                                  metadata["class"].values))

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Parse records.
    dataset = dataset \
        .map(lambda x, y: tf.py_function(
            _dataset_parser,
            [x, y],
            Tout=(tf.float32, tf.int64)),
         num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda x, y: _expand_dims(x, y))

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(buffer_size=200)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset