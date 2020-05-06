import tensorflow as tf
import numpy as np
from .utils import classes_mapping

class EfficientDet:

    def __init__(self, frozen_model):
        """
        Initialize EfficientDet network for inference
        :param frozen_model: path to saved TF model
        See artifacts/efficientdet_export.ipynb for exporting details
        """
        self.net = tf.saved_model.load(frozen_model)

    def predict(self, image):
        """
        :param image: np array of image in BGR format - opencv
        :return: bbox, confidences, classes
        """
        # Define serving type
        infer_function = self.net.signatures["serving_default"]

        # Forward pass
        predictions = infer_function(tf.expand_dims(image, axis=0))
        predictions_numpy = predictions['detections:0'].numpy()

        boxes = predictions_numpy[0][:, 1:5].astype(int)
        # convert [x, y, width, height]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        classes_id = predictions_numpy[0][:, 6].astype(int)
        confidences = predictions_numpy[0][:, 5]

        # Return list of [classes, confidence, bbox]
        results = []
        for c, s, b in zip(classes_id, confidences, boxes):
            one_element = {}
            one_element['label'] = classes_mapping[c]
            one_element['confidence'] = float(s)
            one_element['bbox'] = b.tolist()
            results.append(one_element)

        return results
