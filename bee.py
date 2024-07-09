import torch
import os
from pathlib import Path
import argparse
from PIL import Image

# Function to detect bees in images
def detect_bees(model_path, images_dir, output_dir):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Detect bees in each image
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('jpeg', 'png', 'jpg'))]

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)

        # Perform inference
        results = model(img)

        # Save annotated image with detections
        save_path = os.path.join(output_dir, f'detected_{img_name}')
        results.save(save_path)

        # Display results (optional)
        results.show()

        print(f"Processed: {img_path}. Saved as: {save_path}")

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Bee Detection using YOLOv5')
    parser.add_argument('--model', type=str, default='C:/Users/CIU/Desktop/yolov5/best.pt',
                        help='Path to YOLOv5 model weights file')
    parser.add_argument('--images_dir', type=str, default='C:/Users/CIU/Desktop/yolov5/images',
                        help='Path to directory containing images to detect')
    parser.add_argument('--output_dir', type=str, default='C:/Users/CIU/Desktop/yolov5/output',
                        help='Path to directory where output images with detections will be saved')

    args = parser.parse_args()

    # Run bee detection
    detect_bees(args.model, args.images_dir, args.output_dir)
