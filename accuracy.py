import torch
from pathlib import Path
from tqdm import tqdm
from utils.general import check_img_size, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.plots import plot_one_box

# Set path to your trained YOLOv5 model
weights_path = r'C:\Users\CIU\Desktop\yolov5\best.pt'

# Set path to your dataset images and labels
dataset_path = r'C:\Users\CIU\Desktop\yolov5\images'

# Initialize model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = attempt_load(weights_path, map_location=device)  # load FP32 model
imgsz = check_img_size(640, s=model.stride.max())  # check img_size
model.to(device).eval()

# Initialize dataset
dataset = LoadImages(dataset_path, img_size=imgsz)

# Evaluate each image in the dataset
results = []
for path, img, img0 in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                results.append((path, xyxy, cls.item(), conf.item()))
                
                # Optional: visualize and save results
                plot_one_box(xyxy, img0, label=f'bee', color=(0, 255, 0), line_thickness=3)

# Calculate accuracy metrics (e.g., mAP, precision, recall) based on your needs
# Here, results contains detected objects with their bounding boxes

# Print or save evaluation metrics
print(f"Number of detections: {len(results)}")

# Further processing for accuracy evaluation can be added based on your specific requirements
