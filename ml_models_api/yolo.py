import cv2 as cv2
import numpy as np
from .utils import yolo_mapping, post_process


class Yolov3:

    def __init__(self, weights, config):
        """
        Initialize Yolo network for inference
        :param weights: weights path
        :param graph: graph path
        """
        self.net = cv2.dnn.readNetFromDarknet(config, weights)

        # set opencv backend and CPU inference
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # fixed params for inference
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4  # Non-maximum suppression threshold
        self.input_width = 416  # Width of network's input image
        self.input_height = 416  # Height of network's input image

    def predict(self, image):
        """
        :param image: np array of image in BGR format - opencv
        :return: bbox, confidences, classes
        """
        # Define transformations on the input for Yolo
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.input_width, self.input_height), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Get the output layer in the architecture
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Forward pass
        outs = self.net.forward(output_layers)

        # Postprocessing
        boxes, confidences, classes_id = post_process(image, outs, self.conf_threshold, self.nms_threshold)

        # Return list of [classes, confidence, bbox]
        results = []
        for c, s, b in zip(classes_id, confidences, boxes):
            one_element = {}
            one_element['label'] = yolo_mapping[c]
            one_element['confidence'] = s
            one_element['bbox'] = b
            results.append(one_element)

        return results
