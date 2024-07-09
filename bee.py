import torch
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Path to your trained model
model_path = r'D:\Desktop\yolov5\beetrainedfile.pt'

# Path to directory containing images
images_dir = r'D:\Desktop\yolov5\images'

# Output directory to save annotated images
output_dir = r'D:\Desktop\yolov5\output'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device).eval()

# Function to detect bees and save annotated images
def detect_and_save(image_path, output_dir):
    # Perform inference
    results = model(image_path)

    # Visualize detections and save annotated image
    for img_id, result in enumerate(results.pred):
        # Load original image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Draw boxes and labels
        for box, conf, cls in zip(result[:, :4].tolist(), result[:, 4].tolist(), result[:, 5].tolist()):
            box = [int(b) for b in box]
            xmin, ymin, xmax, ymax = box
            label = f'{model.names[int(cls)]} {conf:.2f}'
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
            draw.text((xmin, ymin), label, fill='red')

        # Save annotated image
        filename = Path(image_path).stem + '_annotated.jpg'
        save_path = Path(output_dir) / filename
        img.save(save_path)
        print(f'Saved annotated image to: {save_path}')

# Function to detect bees on all images in a directory
def detect_on_all_images(images_dir, output_dir):
    # Get all image paths in the directory
    image_paths = list(Path(images_dir).glob('*'))
    
    # Process each image
    for image_path in image_paths:
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f'Detecting bees in {image_path}...')
            detect_and_save(str(image_path), output_dir)

# Example usage:
if __name__ == '__main__':
    detect_on_all_images(images_dir, output_dir)
