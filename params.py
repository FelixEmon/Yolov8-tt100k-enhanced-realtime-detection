from pathlib import Path

IMAGE = 'Image'
VIDEO = 'Video'
SOURCES_LIST = [IMAGE, VIDEO]

IMAGES_DIR = Path('assets/imgs')
DEFAULT_IMAGE = IMAGES_DIR / '1.jpg'
IMAGE_1_PATH = IMAGES_DIR / '1.jpg'
IMAGE_2_PATH = IMAGES_DIR / '2.jpg'
IMAGE_3_PATH = IMAGES_DIR / '3.jpg'
DICT_IMG = {
    'img_1': IMAGE_1_PATH,
    'img_2': IMAGE_2_PATH,
    'img_3': IMAGE_3_PATH
}

VIDEO_DIR = Path('assets/vids')
VIDEO_1_PATH = VIDEO_DIR / '1.mp4'

DICT_VID = {
    'video_1': VIDEO_1_PATH
}

MODEL_DIR = Path('weight/best.pt')