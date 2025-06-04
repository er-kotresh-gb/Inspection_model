import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)

    opt = parser.parse_args()

    model = YOLO(opt.model)
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        workers=opt.workers
    )

if __name__ == "__main__":
    main()
