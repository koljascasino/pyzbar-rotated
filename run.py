import os
from shutil import copyfile

import cv2
import numpy as np
from pyzbar import pyzbar
import structlog

import settings
from mser.mser import find_barcodes

logger = structlog.get_logger(__name__)


def get_file_names(path):
    if settings.DEBUG and settings.DEBUG_IMAGE:
        return [settings.DEBUG_IMAGE]
    else:
        return [f for f in os.listdir(path) if f[-4:].lower() == ".jpg"]


def scale_originals():
    logger.debug("Scaling files.")
    for file_name in get_file_names(settings.PATH_ORIGINAL):
        logger.debug(file_name)
        img = cv2.imread(settings.PATH_ORIGINAL + file_name)
        scaled = cv2.resize(img, (640, 480))
        cv2.imwrite(settings.PATH_SCALED + file_name, scaled)
        copyfile(
            settings.PATH_ORIGINAL + file_name + ".txt",
            settings.PATH_SCALED + file_name + ".txt",
        )


def concat(img1, img2, axis=0):
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


def main():

    # Create scaled version of images if necessary
    directory = os.path.dirname(settings.PATH_SCALED)
    if not os.path.exists(directory):
        logger.debug("No images found. Creating {}.".format(directory))
        os.makedirs(directory)
        scale_originals()

    # Loop over images
    for file_name in get_file_names(settings.PATH_SCALED):

        # Read solution
        with open(settings.PATH_SCALED + file_name + ".txt", "r") as file:
            solution_data = file.read().strip()

        # load image
        img = cv2.imread(settings.PATH_SCALED + file_name)

        # Find list of barcode regions (rotated rectangle) within image
        barcode_rects, debug_img = find_barcodes(img, debug=settings.DEBUG)
        success_rect = None
        invalid_rects_count = 0
        false_positive_data = []
        barcode_images = None

        # Decode barcode regions
        for barcode_rect in barcode_rects:

            # Decode barcode image
            barcode_img = barcode_rect.extract_from(img)
            results = pyzbar.decode(barcode_img)

            # Barcode region cannot be decoded its an invalid barcode region
            if len(results) == 0:
                invalid_rects_count += 1
            for r in results:

                # Check result
                data = r.data.decode("utf-8")
                if data == solution_data:
                    success_rect = barcode_rect
                else:
                    false_positive_data.append(data)

            # Save barcode image
            if settings.DEBUG:
                barcode_images = concat(barcode_images, barcode_img)

        # Log result
        logger.info(
            "Image processed",
            file_name=file_name,
            data=solution_data,
            success=success_rect is not None,
            rect=success_rect.json() if success_rect is not None else None,
            invalid_rects_count=invalid_rects_count,
            false_positive_data=false_positive_data,
        )

        # In debug mode show visualization of detection algorithm
        if settings.DEBUG:
            debug_img = concat(debug_img, barcode_images, axis=1)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", debug_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
