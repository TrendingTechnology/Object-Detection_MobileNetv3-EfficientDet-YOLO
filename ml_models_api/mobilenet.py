import cv2 as cv2
import numpy as np
from .utils import classes_mapping

class MobileNetv3:

    def __init__(self, weights, graph):
        """
        Initialize MobileNet network for inference
        :param weights: weights path
        :param graph: graph path
        """
        self.net = cv2.dnn_DetectionModel(weights, graph)

        # set opencv backend and CPU inference
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # fixed params for inference
        self.conf_threshold = 0.5

    def predict(self, image):
        """
        :param image: np array of image in BGR format - opencv
        :return: bbox, confidences, classes
        """
        # Define transformations on the input for MobileNet
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Forward pass
        classes_id, confidences, boxes = self.net.detect(image, confThreshold=self.conf_threshold)

        # Return list of [classes, confidence, bbox]
        results = []
        for c, s, b in zip(classes_id.flatten(), confidences.flatten(), boxes):
            one_element = {}
            one_element['label'] = classes_mapping[c]
            one_element['confidence'] = float(s)
            one_element['bbox'] = b.tolist()
            results.append(one_element)

        return results
