##
# @file tester.py
# @brief Script to run inference using a trained YOLO segmentation model.
#
# This script loads a trained YOLO model and performs inference on images, videos, or folders.
# It saves the predicted segmentations, bounding boxes, and result texts in the specified output directory.
#
# ### Usage:
# ```bash
# python tester.py --weights best.pt --source path/to/images --save_dir runs/segment
# ```
#
# ### Arguments:
# - '--weights': Path to the trained YOLO model weights (e.g., best.pt).
# - '--source': Path to an image, folder, or video file for performing inference.
# - '--save_dir': Directory where inference results will be saved (default: 'runs/segment').
#
# ### Functions:
# - 'test_model(weights, source, save_dir)': Loads the model and performs inference on the source data, saving the results.
#
# Results include:
# - Segmented masks (if model is segmentation).
# - Bounding boxes.
# - '.txt' annotations in YOLO format.
#
# @author Kotresh GB
# @date 04-06-2025
##


import argparse
from ultralytics import YOLO
import os

def test_model(weights, source, save_dir):
    model = YOLO(weights)
    
    # Inference
    results = model(source, save=True, save_txt=True, project=save_dir, name="inference")

    print(f"\n✅ Inference complete. Results saved to: {os.path.join(save_dir, 'inference')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights (e.g., best.pt)')
    parser.add_argument('--source', type=str, required=True, help='Image, folder or video path for inference')
    parser.add_argument('--save_dir', type=str, default='runs/segment', help='Directory to save results')

    args = parser.parse_args()

    test_model(args.weights, args.source, args.save_dir)
