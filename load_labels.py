import tensorflow as tf


def load_labels(label_file):
    labels = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for label in proto_as_ascii_lines:
        labels.append(label.rstrip())
    return labels
