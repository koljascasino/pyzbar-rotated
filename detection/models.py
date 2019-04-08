import cv2
import numpy as np
import shapely.affinity
import shapely.geometry

from structlog import get_logger

from detection.utils import get_rect

logger = get_logger(__name__)

MARGIN = 5  # Margin around cropped barcode image


class BarcodeRect(object):
    def __init__(
        self,
        center_x,
        center_y,
        width,
        height,
        theta,
        cluster=None,
        box=None,
        predicted_class=None,
        score=None,
    ):
        if box is None:
            angle = theta / np.pi * 180
            box = np.round(
                np.array(cv2.boxPoints(((center_x, center_y), (width, height), angle)))
            )
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.theta = theta
        self.box = box
        self.cluster = cluster
        self.predicted_class = predicted_class
        self.score = score

    @classmethod
    def from_coords(cls, coords, cluster=None):
        box = get_rect(coords, short_side_is_width=False)
        return BarcodeRect(
            center_x=box[0][0],
            center_y=box[0][1],
            width=box[1, 0],
            height=box[1, 1],
            theta=box[2, 0],
            cluster=cluster,
            box=box[3:].astype(int),
        )

    @classmethod
    def from_mask(cls, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero = cv2.findNonZero(gray)
        return cls.from_coords(non_zero)

    def get_contour(self):
        c = shapely.geometry.box(
            -self.width / 2.0, -self.height / 2.0, self.width / 2.0, self.height / 2.0
        )
        rc = shapely.affinity.rotate(c, self.theta, use_radians=True)
        return shapely.affinity.translate(rc, self.center_x, self.center_y)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

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
        angle = self.theta / np.pi * 180
        matrix = cv2.getRotationMatrix2D(
            center=(self.center_x, self.center_y), angle=angle, scale=1
        )
        rotated = cv2.warpAffine(src=img, M=matrix, dsize=shape)

        width = np.ceil(self.width).astype(int) + MARGIN
        height = np.ceil(self.height).astype(int) + MARGIN
        x = int(self.center_x - width / 2)
        y = int(self.center_y - height / 2)
        x = min(img.shape[1], max(0, x))
        y = min(img.shape[0], max(0, y))

        cropped = rotated[y : y + height, x : x + width]

        return cropped

    def get_mask(self, img):
        """Returns a binary mask of the barcode in an image with the same shape as the original image. """
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [self.box.astype(np.int32)], 0, (255, 255, 255), -1)
        return mask

    def draw(self, img):
        """Returns a binary mask of the barcode in an image with the same shape as the original image. """
        cv2.drawContours(img, [self.box.astype(np.int32)], 0, (0, 255, 0), 2)
        return img


class BoundingBox(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @classmethod
    def from_coords(cls, coords):
        x, y, w, h = cv2.boundingRect(coords)
        return BoundingBox(x, y, w, h)

    @classmethod
    def from_mask(cls, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        non_zero = cv2.findNonZero(gray)
        return cls.from_coords(non_zero)

    def draw(self, img):
        cv2.rectangle(
            img,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (0, 0, 255),
            2,
        )
        return img

    @property
    def rectangle(self):
        return (self.x, self.y), (self.x + self.width, self.y + self.height)

    def __str__(self):
        return "%i,%i,%i,%i" % (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )
