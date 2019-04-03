from math import cos, sin
from collections import namedtuple
from structlog import get_logger

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

logger = get_logger()

MARGIN = 5  # Margin around cropped barcode image
np.random.seed(0)

colors = np.array(
    [
        [43, 43, 200],
        [43, 106, 200],
        [43, 169, 200],
        [43, 200, 163],
        [43, 200, 101],
        [54, 200, 43],
        [116, 200, 43],
        [179, 200, 43],
        [200, 153, 43],
        [200, 90, 43],
        [200, 43, 64],
        [200, 43, 127],
        [200, 43, 190],
        [142, 43, 200],
        [80, 43, 200],
    ]
)

Cluster = namedtuple("Cluster", ["number", "count", "color"])


class BarcodeRect(object):
    def __init__(self, center_x, center_y, width, height, theta, cluster, box):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.theta = theta
        self.cluster = cluster
        self.box = box

    @classmethod
    def from_coords(cls, coords, cluster):
        box = _get_bounding_box(coords)
        return BarcodeRect(
            center_x=box[0][0],
            center_y=box[0][1],
            width=box[1, 0],
            height=box[1, 1],
            theta=box[2, 0],
            cluster=cluster,
            box=box[3:].astype(int),
        )

    @property
    def area(self):
        return self.width * self.height

    def json(self):
        return {
            "center_x": self.center_y,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "theta": self.theta,
        }

    def extract_from(self, img):
        """Extract barcode image from original image where barcode was found

        Rotate original image around center with angle theta (in deg)
        then crop the image according to width and height of the barcode
        """
        shape = (max(img.shape), max(img.shape))
        matrix = cv2.getRotationMatrix2D(
            center=(self.center_x, self.center_y), angle=self.theta, scale=1
        )
        rotated = cv2.warpAffine(src=img, M=matrix, dsize=shape)

        width = self.width + MARGIN
        height = self.height + MARGIN
        x = int(self.center_x - width / 2)
        y = int(self.center_y - height / 2)
        x = min(img.shape[1], max(0, x))
        y = min(img.shape[0], max(0, y))

        cropped = rotated[y : y + height, x : x + width]

        return cropped


def _bgr2rgb(bgr):
    return [float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0]


def _get_bounding_box(coords):
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


def _get_color_dict(clusters):
    """Return a dictionary of random BGR colors for each cluster."""
    cluster_colors_bgr = colors[np.random.choice(len(colors), clusters.shape)]
    return dict(zip(clusters, cluster_colors_bgr))


def _plot_clustering_space(x, y, clusters, cluster_idx):
    """Show a matplotlib plot of the clustering space."""
    color_dict = {}
    for cluster in clusters:
        color_dict[cluster.number] = cluster.color

    cluster_colors_rgb = [_bgr2rgb(color_dict[c]) for c in cluster_idx]
    plt.scatter(x, y, c=cluster_colors_rgb, linewidths=0)
    plt.show()


def _draw_final_boxes(img, results):
    """Draw bounding box around cluster of bars."""
    for rect in results:
        cv2.drawContours(img, [rect.box], 0, (0, 0, 255), 1)


def _colorize_clusters(img, blobs, clusters, cluster_idx):
    """Colorize bars of each cluster to visualize what clustering algorithm detected."""

    # Convert background image to grayscale to better highlight clusters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for idx, cluster in enumerate(clusters):
        cluster = clusters[idx]
        bar_blobs = blobs[np.where(cluster_idx == cluster.number)]
        for coords in bar_blobs:
            x = coords[:, 0]
            y = coords[:, 1]
            img[y, x] = cluster.color

    return img


def _cluster_bars(bar_boxes, img_size, plot=False):
    """Cluster bars in (theta, height, center_x, center_y) space with appropriate scaling."""
    center_x = bar_boxes[:, 0, 0] / img_size
    center_y = bar_boxes[:, 0, 1] / img_size
    heights = bar_boxes[:, 1, 1] / img_size
    thetas = bar_boxes[:, 2, 0]
    X = np.transpose([thetas, heights, center_x, center_y])

    # Run clustering
    dbscan = DBSCAN(min_samples=5, eps=0.1).fit(X)
    cluster_idx = dbscan.labels_
    numbers, counts = np.unique(cluster_idx, return_counts=True)

    # Create dictionary with cluster info
    clusters = []
    for number, count in zip(numbers, counts):
        color = [0, 0, 0] if number == -1 else colors[np.random.choice(len(colors))]
        clusters.append(Cluster(number, count, color=color))

    if plot:
        _plot_clustering_space(thetas, heights, clusters, cluster_idx)

    return clusters, cluster_idx


def find_barcodes(img, debug=False):
    """Find barcodes within image and return a list of extracted barcode images."""

    # Run MSER on gray scale image to detect bars
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(50)
    blobs, _ = mser.detectRegions(gray)
    blobs = np.array(blobs)

    # Calculate rotated bounding box around each blob detected by mser
    bar_boxes = np.zeros((blobs.shape[0], 7, 2))
    for idx in range(blobs.shape[0]):
        coords = blobs[idx]
        box = _get_bounding_box(coords)
        bar_boxes[idx, :, :] = box

    # Only consider blobs where height ratio > 10 (otherwise its definitely not a bar)
    with np.errstate(divide="ignore"):
        filter_height_ratio = np.where(bar_boxes[:, 1, 1] / bar_boxes[:, 1, 0] > 10)
    blobs = blobs[filter_height_ratio]
    bar_boxes = bar_boxes[filter_height_ratio]

    # No bars found
    if len(bar_boxes) == 0:
        logger.debug("No bars found in image.")
        return [], img

    # Cluster bars in (theta, height, center_x, center_y) space with appropriate scaling
    clusters, cluster_idx = _cluster_bars(bar_boxes, max(img.shape), plot=debug)

    # Construct rotated bounding box around clusters of bars
    results = []
    for idx, cluster in enumerate(clusters):
        if cluster.number == -1:
            continue
        coords = bar_boxes[np.where(cluster_idx == cluster.number), 3:, :]
        coords = coords.reshape(-1, 2).astype(int)
        results.append(BarcodeRect.from_coords(coords, cluster))

    # Colorize clusters
    if debug:
        img = _colorize_clusters(img, blobs, clusters, cluster_idx)

    # Run post processing
    results = post_processing(results, bar_boxes, cluster_idx)

    # Draw box around barcodes identifed
    if debug:
        _draw_final_boxes(img, results)

    return results, img


def combine_overlapping_areas(results, cluster_idx):
    """Combine overlapping or adjacent barcode areas.

    Recursively try combining all pairs of barcode areas in O(N^2). If the combined area of two barcodes is less than
    or little more than the sum of the two combined areas, then do combine them.
    """
    for i in range(len(results)):
        rect = results[i]
        for j in range(i + 1, len(results)):
            rect2 = results[j]
            combined = BarcodeRect.from_coords(
                np.vstack((rect.box, rect2.box)), rect.cluster
            )
            if combined.area <= (rect.area + rect2.area) * 1.05:
                results[i] = combined
                cluster_idx[
                    np.where(cluster_idx == rect2.cluster.number)
                ] = rect.cluster.number
                return combine_overlapping_areas(
                    results[:j] + results[j + 1 :], cluster_idx
                )

    return results


def post_processing(results, bar_boxes, cluster_idx):
    """Post process identified barcode regions, combine overlapping or adjacent areas."""
    combined = combine_overlapping_areas(results, cluster_idx)
    filtered = []
    for rect in combined:

        # Barcode should be a horizontal series of bars, not a a vertical stack
        # i.e barcode box should have 90° different orientation than bars
        # If this is not the case ignore the candidate
        mean_bar_theta = np.mean(
            bar_boxes[np.where(cluster_idx == rect.cluster.number), 2, 0]
        )
        if abs(rect.theta - mean_bar_theta) < np.pi / 4:
            continue

        # Convert angle from perpendicular middle line (in radians) to angle (in degrees)
        # Switch width and height
        filtered.append(
            BarcodeRect(
                center_x=rect.center_x,
                center_y=rect.center_y,
                width=np.ceil(rect.height).astype(int),
                height=np.ceil(rect.width).astype(int),
                theta=(rect.theta * 180 / np.pi + 180) % 180 - 90,
                cluster=rect.cluster,
                box=rect.box,
            )
        )

    return filtered
