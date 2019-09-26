import argparse

import numpy as np
import tensorflow as tf

from load_model import load_model
from load_labels import load_labels
from read_image import read_tensor_from_image_file


if __name__ == "__main__":
    file_name = ""
    model_file = "output/graph.pb"
    label_file = "output/labels.txt"
    input_layer = "Placeholder"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    args = parser.parse_args()

    if args.image:
        file_name = args.image

    graph = load_model(model_file)
    tensor = read_tensor_from_image_file(file_name)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    print("Guess:")
    for i in top_k:
        print(str(labels[i]) + " = " + str(results[i]))
