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
