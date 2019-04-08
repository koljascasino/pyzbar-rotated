from math import cos, sin

import cv2
import numpy as np


def get_rect(coords):
    """Calculate minimum rotated rectangle containing list of points"""
    rect = cv2.minAreaRect(coords)
    center_x = rect[0][0]
    center_y = rect[0][1]
    width = rect[1][0]
    height = rect[1][1]
    angle = (rect[2] + 90) % 180 - 90

    # Switch height and width if width > height
    if width > height:
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
