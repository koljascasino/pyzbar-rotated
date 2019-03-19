import logging.config

import structlog

# Directory where test images can be found
# Download from http://artelab.dista.uninsubria.it/downloads/datasets/barcode/medium_barcode_1d/medium_barcode_1d.html
# If you use this dataset in your work, please add the following reference:
# Neural Image Restoration For Decoding 1-D Barcodes Using Common Camera Phones
# Alessandro Zamberletti, Ignazio Gallo, Moreno Carullo and Elisabetta Binaghi
# Computer Vision, Imaging and Computer Graphics. Theory and Applications, Springer Berlin Heidelberg, 2011
PATH_ORIGINAL = "./data/BarcodeDataSets/Dataset1/"
PATH_SCALED = "./data/BarcodeDataSets/Dataset1/scaled/"

# Use this flag to show picture of how MSER algorithm detects regions of bars and and clusters them
DEBUG = False
DEBUG_IMAGE = "PICT0013.JPG"

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
    "root": {"handlers": ["console", "file"], "level": logging.DEBUG},
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
