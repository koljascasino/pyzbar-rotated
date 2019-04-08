import os
from math import cos, sin

import cv2
import numpy as np

import settings


def get_dataset():
    """Return list of file names in dataset directory."""
    if settings.SHOW_IMAGE and settings.IMAGE:
        return [settings.IMAGE]
    return os.listdir(settings.PATH)


def get_rect(coords, short_side_is_width=True):
    """Calculate minimum rotated rectangle containing list of points"""
    rect = cv2.minAreaRect(coords)
    center_x = rect[0][0]
    center_y = rect[0][1]
    width = rect[1][0]
    height = rect[1][1]
    angle = (rect[2] + 90) % 180 - 90

    # Switch height and width if width > height
    if (height > width) ^ short_side_is_width:
        width = rect[1][1]
        height = rect[1][0]
        angle = rect[2] % 180 - 90

    # Parametrize perpendicular middle line with (theta, rho)
    # theta = angle of perpendicular middle line in radians
    # rho = distance from origin to perpendicular middle line (Hough transform)
    theta = np.pi * angle / 180
    rho = center_x * cos(theta) + center_y * sin(theta)

    # Corners of rotated bounding box
    box = cv2.boxPoints(rect)

    return np.vstack(
        (
            np.array([[center_x, center_y], [width, height], [theta, rho]]),
            np.round(np.array(box)),
        )
    )


def img_concat(img1, img2, axis=0):
    """Concatenate two images along their height or width. Pad if images do not have the same shape."""
    assert axis in (0, 1)

    if img1 is None:
        return img2
    if img2 is None:
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if axis == 0:
        vis = np.zeros((h1 + h2, max(w1, w2), img1.shape[2]), np.uint8)
        vis[:h1, :w1, :] = img1
        vis[h1 : h1 + h2, :w2, :] = img2

    else:
        vis = np.zeros((max(h1, h2), w1 + w2, img1.shape[2]), np.uint8)
        vis[:h1, :w1, :] = img1
        vis[:h2, w1 : w1 + w2, :] = img2

    return vis


def fix_label():
    """Fix incorrect ground truth masks in artelab dataset."""
    file_name = "05102009082.png"
    ty = 12
    tx = 59
    ground_truth_mask = cv2.imread(settings.PATH_GT_MASKS + file_name)
    ground_truth_mask[ty:, tx:, :] = ground_truth_mask[:-ty, :-tx, :]
    cv2.imwrite(settings.PATH_GT_MASKS + file_name, ground_truth_mask)
