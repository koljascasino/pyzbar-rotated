import os

import cv2
import numpy as np
from pyzbar import pyzbar
import structlog

import settings
from mser.mser import find_barcodes

logger = structlog.get_logger(__name__)


def main():
    """Decode the 1D barcodes in a set of images and log the result."""
    for file_name in _get_file_names(settings.PATH):

        # load image and detection mask
        img = cv2.imread(settings.PATH + file_name)
        ground_truth_mask = cv2.imread(settings.PATH_GROUND_TRUTH + file_name)

        # Find list of barcode regions (rotated rectangle) within image
        barcode_regions, debug_img = find_barcodes(img)
        barcode_regions_mask = np.zeros(img.shape, np.uint8)
        barcode_images = None
        result = []

        # Decode barcode regions
        for barcode_region in barcode_regions:

            # Decode barcode image
            barcode_img = barcode_region.extract_from(img)
            barcode_mask = barcode_region.get_mask(img)
            decoded = pyzbar.decode(barcode_img)

            # Combine masks from multiple regions
            barcode_regions_mask += barcode_mask

            # Keep result for logging
            data = ", ".join([d.data.decode("utf-8") for d in decoded])
            result.append({"data": data, "region": barcode_region.json()})

            if settings.SHOW_VISUAL:
                barcode_images = _img_concat(barcode_images, barcode_img)

        # Jaccard_accuracy = intersection over union of the two binary masks
        r = barcode_regions_mask.max(axis=-1).astype(bool)
        u = ground_truth_mask.max(axis=-1).astype(bool)
        jaccard_accuracy = float((r & u).sum()) / (r | u).sum()

        # Calculate bounding box of ground truth
        bbox = BoundingBox.from_mask(ground_truth_mask)

        # Log result
        logger.info(
            "Image processed",
            file_name=file_name,
            jaccard_accuracy=jaccard_accuracy,
            success=jaccard_accuracy > 0.5,
            result=result,
            ground_truth_bbox=bbox.json(),
        )

        # In debug mode show visualization of detection algorithm
        if settings.SHOW_VISUAL:
            error_img = np.zeros(img.shape, np.uint8)
            error_img[r & u] = np.array([255, 255, 255], dtype=np.uint8)
            error_img[np.logical_xor(r, u)] = np.array([0, 0, 255], dtype=np.uint8)
            debug_img = cv2.addWeighted(debug_img, 0.6, error_img, 0.4, 0)

            # Draw bounding box of ground truth barcode area
            cv2.rectangle(debug_img, *bbox.rectangle, (0, 0, 255), 2)

            # Append barcode pictures
            debug_img = _img_concat(debug_img, barcode_images, axis=1)

            # Show visualization
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", debug_img)
            cv2.waitKey(0)


def _get_file_names(path):
    """Get the list of file names with barcode images to decode."""
    if settings.SHOW_VISUAL and settings.DEBUG_IMAGE:
        return [settings.DEBUG_IMAGE]
    else:
        return os.listdir(path)


def _img_concat(img1, img2, axis=0):
    """Concatenate two images along their height or width. Pad if images do not have the same shape."""
    assert axis in (0, 1)

    if img1 is None:
        return img2
    if img2 is None:
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if axis == 0:
        vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
        vis[:h1, :w1, :3] = img1
        vis[h1 : h1 + h2, :w2, :3] = img2

    else:
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = img1
        vis[:h2, w1 : w1 + w2, :3] = img2

    return vis


def fix_label():
    file_name = "05102009082.png"
    ty = 12
    tx = 59
    ground_truth_mask = cv2.imread(settings.PATH_GROUND_TRUTH + file_name)
    ground_truth_mask[ty:, tx:, :] = ground_truth_mask[:-ty, :-tx, :]
    cv2.imwrite(settings.PATH_GROUND_TRUTH + file_name, ground_truth_mask)


class BoundingBox(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @classmethod
    def from_mask(cls, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(non_zero)
        return BoundingBox(x, y, w, h)

    @property
    def rectangle(self):
        return (self.x, self.y), (self.x + self.width, self.y + self.height)

    def json(self):
        center_x = self.x + self.width / 2.0
        center_y = self.x + self.height / 2.0
        return {
            "center_x": center_x,
            "center_y": center_y,
            "width": self.width,
            "height": self.height,
        }


if __name__ == "__main__":
    main()
