from timeit import default_timer as timer

import cv2
import numpy as np
import structlog
from pyzbar import pyzbar

import settings
from detection.mser import find_barcodes
from detection.utils import get_dataset, img_concat

logger = structlog.get_logger(__name__)


def main():
    """Decode the 1D barcodes in a set of images and log the result."""
    stats = []
    start = timer()

    for file_name in get_dataset():

        # load image and ground truth detection mask
        img = cv2.imread(settings.PATH + file_name)
        ground_truth_mask = cv2.imread(settings.PATH_GT_MASKS + file_name)

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
            debug_img = barcode_region.draw(debug_img)

            # Combine masks from multiple detected regions
            barcode_regions_mask += barcode_mask

            # Decode barcode
            decoded = pyzbar.decode(barcode_img)

            # Keep result for logging
            data = ", ".join([d.data.decode("utf-8") for d in decoded])
            result.append({"data": data, "region": barcode_region.json()})

            if settings.SHOW_IMAGE:
                barcode_images = img_concat(barcode_images, barcode_img)

        # Jaccard_accuracy = intersection over union of the two binary masks
        jaccard_accuracy = 0
        if ground_truth_mask is not None:
            r = barcode_regions_mask.max(axis=-1).astype(bool)
            u = ground_truth_mask.max(axis=-1).astype(bool)
            jaccard_accuracy = float((r & u).sum()) / (r | u).sum()
            stats.append(jaccard_accuracy)

        # Log result
        logger.info(
            "Image processed",
            file_name=file_name,
            jaccard_accuracy=jaccard_accuracy,
            success=jaccard_accuracy > 0.5,
            result=result,
        )

        # In debug mode show visualization of detection algorithm
        if settings.SHOW_IMAGE:

            # Add alpha channel
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2BGRA)
            if barcode_images is not None:
                barcode_images = cv2.cvtColor(barcode_images, cv2.COLOR_BGR2BGRA)

            # Overlay error mask
            # Pixel-wise difference between ground truth and detected barcodes
            if ground_truth_mask is not None:
                error_img = np.zeros(debug_img.shape, np.uint8)
                error_img[r & u] = np.array([0, 0, 0, 0], dtype=np.uint8)
                error_img[np.logical_xor(r, u)] = np.array(
                    [0, 0, 255, 1], dtype=np.uint8
                )
                debug_img = cv2.addWeighted(debug_img, 1, error_img, 0.5, 0)

            # Append barcode pictures to the right
            debug_img = img_concat(debug_img, barcode_images, axis=1)

            # Show visualization
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", debug_img)
            cv2.waitKey(0)

    # Calculate final stats
    end = timer()
    accuracy = np.array(stats).mean()
    successes = np.where(np.array(stats) > 0.5)[0]
    logger.info(
        "Final stats",
        accuracy=accuracy,
        detection_rate=float(len(successes)) / len(stats),
        fps=len(stats) / (end - start),
    )


if __name__ == "__main__":
    main()
