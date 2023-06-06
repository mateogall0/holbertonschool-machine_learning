#!/usr/bin/env python3
"""
Initialize Yolo
"""


import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Yolo class
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes_file:
            self.class_names = [
                line.strip() for line in classes_file.readlines()
                ]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors


    def process_outputs(self, outputs, image_size):
        """
        outputs -- list of numpy.ndarrays containing the predictions from the
        Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size -- numpy.ndarray containing the image’s original size
        [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box's class
            probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, _, _ = output.shape
            box = np.zeros_like(output[..., :4])
            box[..., 0] = (1 / (1 + np.exp(-output[..., 0])) + np.arange(grid_width)[np.newaxis, np.newaxis, :, np.newaxis])
            box[..., 1] = (1 / (1 + np.exp(-output[..., 1])) + np.arange(grid_height)[np.newaxis, :, np.newaxis])
            box[..., 2] = np.exp(output[..., 2]) * self.anchors[:, 0][np.newaxis, np.newaxis, :, 0]
            box[..., 3] = np.exp(output[..., 3]) * self.anchors[:, 1][np.newaxis, np.newaxis, :, 1]
            box[..., 0:2] /= np.array([grid_width, grid_height])
            box[..., 2:4] /= np.array([self.model.input.shape[1].value, self.model.input.shape[2].value])
            box[..., 0] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 2] *= image_size[1]
            box[..., 3] *= image_size[0]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
