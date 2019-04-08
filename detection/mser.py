from collections import namedtuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from structlog import get_logger

import settings
from detection.models import BarcodeRect
from detection.utils import get_rect

logger = get_logger(__name__)

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


def _bgr2rgb(bgr):
    return [float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0]


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


def _cluster_bars(bar_boxes, img_size):
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

    if settings.PLOT_CLUSTERING_SPACE:
        _plot_clustering_space(thetas, heights, clusters, cluster_idx)

    return clusters, cluster_idx


def find_barcodes(img):
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
        box = get_rect(coords)
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
    clusters, cluster_idx = _cluster_bars(bar_boxes, max(img.shape))

    # Construct rotated bounding box around clusters of bars
    results = []
    for idx, cluster in enumerate(clusters):
        if cluster.number == -1:
            continue
        coords = bar_boxes[np.where(cluster_idx == cluster.number), 3:, :]
        coords = coords.reshape(-1, 2).astype(int)
        results.append(BarcodeRect.from_coords(coords, cluster))

    # Colorize clusters
    if settings.COLORIZE_CLUSTERS:
        img = _colorize_clusters(img, blobs, clusters, cluster_idx)

    # Run post processing
    results = post_processing(results, bar_boxes, cluster_idx)

    return results, img


def combine_overlapping_areas(results, cluster_idx):
    """Combine overlapping or adjacent barcode areas.

    Recursively try combining all pairs of barcode areas in O(N^2). If the combined area of two barcodes is less than
    or little more than the sum of the two combined areas, then do combine them.
    """
    for i in range(len(results)):
        this = results[i]
        for j in range(i + 1, len(results)):
            other = results[j]
            intersection = this.intersection(other)
            union = BarcodeRect.from_coords(
                np.vstack((this.box, other.box)), this.cluster
            )
            if intersection.area > 0 or union.area <= (this.area + other.area) * 1.05:
                results[i] = union
                cluster_idx[
                    np.where(cluster_idx == other.cluster.number)
                ] = this.cluster.number
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
            filtered.append(rect)

    return filtered
