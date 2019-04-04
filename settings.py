import logging.config

import structlog

# ArTe-Lab Rotated Barcode Database (368 images)
# Download extended 1d_barcode_extended_plain.zip from
# http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html
# If you use this dataset in your work, please add the following reference:
# Robust Angle Invariant 1D Barcode Detection
# Alessandro Zamberletti, Ignazio Gallo and Simone Albertini
# Proceedings of the 2nd Asian Conference on Pattern Recognition (ACPR), Okinawa, Japan, 2013
PATH = "./data/1d_barcode_extended_plain/Original/"
PATH_GROUND_TRUTH = "./data/1d_barcode_extended_plain/Detection/"

# Use this flag to show picture of how MSER algorithm detects regions of bars and and clusters them
DEBUG_IMAGE = None  # "05102009082.png"
SHOW_VISUAL = False
PLOT_CLUSTERING_SPACE = False

# Logger config
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "level": logging.INFO,
            "filename": "run.log",
        },
        "console": {"class": "logging.StreamHandler", "level": logging.DEBUG},
    },
    "loggers": {
        "__main__": {
            "handlers": ["console", "file"],
            "level": logging.DEBUG,
            "propagate": False,
        },
        "mser": {"handlers": ["console"], "level": logging.DEBUG, "propagate": False},
    },
    "root": {"handlers": ["console", "file"], "level": logging.WARNING},
}
logging.config.dictConfig(LOGGING_CONFIG)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
