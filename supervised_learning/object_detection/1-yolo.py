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
        ih, iw = image_size

        for i, output in enumerate(outputs):
            # Get dimensions of the current output
            gh, gw, anchorBoxes, _ = output.shape

            # Create an empty array to store processed boundary boxes
            box = np.zeros(output[:, :, :, :4].shape)

            # Extract predicted coordinates and dimensions of boundary boxes
            tx = output[:, :, :, 0]
            ty = output[:, :, :, 1]
            tw = output[:, :, :, 2]
            th = output[:, :, :, 3]

            # Get anchor box widths and heights for each grid cell
            pwTotal = self.anchors[:, :, 0]
            phTotal = self.anchors[:, :, 1]

            # Reshape anchor box widths and heights to match grid dimensions
            pw = np.tile(pwTotal[i], gw).reshape(gw, 1, len(pwTotal[i]))
            ph = np.tile(phTotal[i], gh).reshape(gh, 1, len(phTotal[i]))

            # Create grid coordinates for positioning boundary boxes
            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gw),
                         gh).reshape(gh, gh).T.reshape(gh, gh, 1)

            # Compute absolute coordinates and dimensions of boundary boxes
            bx = (1 / (1 + np.exp(-tx)) + cx) / gw
            by = (1 / (1 + np.exp(-ty)) + cy) / gh
            bw = (np.exp(tw) * pw) / self.model.input.shape[1].value
            bh = (np.exp(th) * ph) / self.model.input.shape[2].value

            # Update box array with computed box coordinates and dimensions
            box[:, :, :, 0] = (bx - (bw / 2)) * iw
            box[:, :, :, 1] = (by - (bh / 2)) * ih
            box[:, :, :, 2] = (bx + (bw / 2)) * iw
            box[:, :, :, 3] = (by + (bh / 2)) * ih
            boxes.append(box)

            # Compute box confidences and reshape to match grid dimensions
            temp = output[:, :, :, 4]
            sigmoid = (1 / (1 + np.exp(-temp)))
            box_confidences.append(sigmoid.reshape(gh, gw, anchorBoxes, 1))

            # Compute box class probabilities
            temp = output[:, :, :, 5:]
            box_class_probs.append((1 / (1 + np.exp(-temp))))

        return boxes, box_confidences, box_class_probs
