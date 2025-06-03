import os
import cv2
import albumentations as A
from glob import glob
from concurrent.futures import ThreadPoolExecutor

class PolygonAugmentor:
    def __init__(self, image_dir, label_dir, output_img_dir, output_lbl_dir, logger, num_augmentations=3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_img_dir = output_img_dir
        self.output_lbl_dir = output_lbl_dir
        self.num_augmentations = num_augmentations
        self.logger = logger

        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_lbl_dir, exist_ok=True)

        self.logger.info(f"Initialized PolygonAugmentor.")
        self.logger.info(f"Image Dir: {self.image_dir}")
        self.logger.info(f"Label Dir: {self.label_dir}")
        self.logger.info(f"Output Image Dir: {self.output_img_dir}")
        self.logger.info(f"Output Label Dir: {self.output_lbl_dir}")

        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(p=0.3)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def read_polygons(self, label_path):
        polygons = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    polygons.append((class_id, points))
            # self.logger.info(f"Read polygons from {label_path}")
        except Exception as e:
            self.logger.error(f"Error reading {label_path}: {e}")
        return polygons

    def write_polygons(self, polygons, output_path):
        try:
            with open(output_path, 'w') as f:
                for class_id, points in polygons:
                    coords_flat = []
                    for x, y in points:
                        x = min(max(x, 0.0), 1.0)
                        y = min(max(y, 0.0), 1.0)
                        coords_flat.extend([x, y])
                    line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords_flat)
                    f.write(line + "\n")
            # self.logger.info(f"Saved augmented label: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save label {output_path}: {e}")

    def process_image(self, image_path):
        name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.label_dir, f'{name}.txt')

        if not os.path.exists(label_path):
            self.logger.warning(f"Label not found for image: {name}")
            return

        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not read image: {image_path}")
            return

        polygons = self.read_polygons(label_path)
        height, width = image.shape[:2]

        for i in range(self.num_augmentations):
            keypoints = []
            for class_id, points in polygons:
                for (x, y) in points:
                    keypoints.append((x * width, y * height))

            try:
                augmented = self.augment(image=image, keypoints=keypoints)
            except Exception as e:
                self.logger.error(f"Augmentation failed for {name}, iteration {i}: {e}")
                continue

            aug_img = augmented['image']
            aug_keypoints = augmented['keypoints']

            polygons_aug = []
            idx = 0
            for class_id, points in polygons:
                num_points = len(points)
                pts = aug_keypoints[idx:idx+num_points]
                idx += num_points
                norm_pts = [(x / width, y / height) for (x, y) in pts]
                polygons_aug.append((class_id, norm_pts))

            new_name = f'{name}_aug{i}'
            out_img_path = os.path.join(self.output_img_dir, f'{new_name}.jpg')
            out_lbl_path = os.path.join(self.output_lbl_dir, f'{new_name}.txt')

            try:
                cv2.imwrite(out_img_path, aug_img)
                self.write_polygons(polygons_aug, out_lbl_path)
                # self.logger.info(f"Saved: {new_name}")
            except Exception as e:
                self.logger.error(f"Error saving files for {new_name}: {e}")

    def run(self):
        images = glob(os.path.join(self.image_dir, '*.jpg'))
        if not images:
            self.logger.warning("No images found to process.")
            return
        self.logger.info(f"Found {len(images)} images to process.")
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_image, images)
        self.logger.info("Augmentation completed.")
