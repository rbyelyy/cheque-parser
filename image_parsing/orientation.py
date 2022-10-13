import commands
import cv2
import logging.config
import os
import re
import sys
from PIL import Image


# create logger
logger = logging.getLogger(__name__)
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
})


def rotate(image_path, degree):
    """
    Rotate image based on degree
    """
    try:
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(image_path, rotated)
        logger.info("File {0} have been rotated on {1} degree".format(image_path, degree))
    except AttributeError:
        exit("Cannot open {0} file".format(image_path))

def text_aligment(image_path):
    """
    Get text alignment from tesseract
    """
    if os.path.isfile(image_path):
        # get information from tesseract
        tesseract_output = commands.getstatusoutput("tesseract " + image_path + " image_result -psm 0")
        if 'Skipping this page' in tesseract_output[1]:
        	logger.error("Cannot contninue as far as following error occurs:\n {0}".format(tesseract_output[1]))
        	logger.info("Trying to applay static rotation ...")
        	static_orientation(image_path)
        	exit("End of rotation process...")
        else:
	        # get pattern with orientation degree
	        orientation = [pattern for pattern in tesseract_output[1].split('\n') if 'Orientation in degrees' in pattern]
	        # grab degree from the text
	        degree = re.sub("[^0-9]", "", orientation[0])
	        logger.info("Text on the image {0} have been sucesfully parsed".format(image_path))
	        return degree
    else:
        exit("File {0} does not exists".format(image_path))

def static_orientation(image_path):
    """ 
    This function autorotates a picture 
    """
    image = Image.open(image_path)
    wid,heig=image.size
    print image.size
    if wid > heig:
        graus = 270
        image = image.rotate(graus)
        image.save("osd_"+image_path, quality=100)
        logger.info("Image {0} have been staticaly rotated".format(image_path))
    else:
    	logger.info("Image {0} have not been staticaly rotated".format(image_path))


if __name__ == "__main__":
	image_path = sys.argv[1]
	degree = int(text_aligment(image_path))
	rotate(image_path, -degree)