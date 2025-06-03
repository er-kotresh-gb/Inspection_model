import os
import cv2
import yaml
import logging
from pathlib import Path

def setup_logger(name='yolo_logger', level=logging.INFO):
    """Setup a basic logger."""
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logger()

def read_yaml(path):
    """Read a YAML file."""
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load YAML file: {path}. Error: {e}")
        return None

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image(path):
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        logger.warning(f"Could not read image: {path}")
    return img

def save_image(path, image):
    """Save image to disk."""
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")

def load_labels(label_path):
    """Load YOLO-format labels from a .txt file."""
    bboxes, class_labels = [], []
    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    """Save YOLO-format labels to a .txt file."""
    try:
        with open(label_path, 'w') as f:
            for cls, box in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(map(str, box))}\n")
    except Exception as e:
        logger.error(f"Error saving label file {label_path}: {e}")
