import os
import sys

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Recommender:
    doc_weights, ids = None, None

    def __init__(self, model_file, id_file):
        # LOAD MODEL AND LABELS
        model = tf.keras.models.load_model(model_file)
        self.ids = np.load(id_file, allow_pickle=True)
        doc_layer = model.get_layer("doc_embedding")
        self.doc_weights = doc_layer.get_weights()[0]
        self.doc_weights = self.doc_weights / np.linalg.norm(
            self.doc_weights, axis=1
        ).reshape((-1, 1))

    def run(self, doc) -> list:
        if doc not in self.ids.values:
            print("Cannot find document!")
            return []

        doc_num = self.ids[self.ids == doc].index[0]

        # PREDICTION
        dists = np.dot(self.doc_weights, self.doc_weights[doc_num])
        sorted_ids = np.argsort(dists)
        sorted_dists = sorted(dists)[-11:]
        closest_ids = sorted_ids[-11:]

        result = []
        for i in range(len(closest_ids)):
            result.append(
                (self.ids.values[closest_ids[i]], sorted_dists[i].astype(float))
            )

        result.reverse()
        return result


if __name__ == "__main__":
    modelFile = sys.argv[1]
    idFile = sys.argv[2]
    document = sys.argv[3]

    r = Recommender(modelFile, idFile)
    for r in r.run(document):
        print(r)
