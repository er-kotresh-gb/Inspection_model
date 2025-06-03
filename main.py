import logging
from src.augmentor import PolygonAugmentor

if __name__ == "__main__":
    # Create a logger
    logger = logging.getLogger("PolygonAugmentor")
    logger.setLevel(logging.INFO)

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to logger
    logger.addHandler(ch)

    augmentor = PolygonAugmentor(
        image_dir='data/Images/train/images',
        label_dir='data/Images/train/labels',
        output_img_dir='data/Images/train/images',
        output_lbl_dir='data/Images/train/labels',
        num_augmentations=3,
        logger=logger
    )
    augmentor.run()
    print("Augmentation completed.")
