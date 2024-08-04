import torch
from ultralytics import YOLO
import os

def train_yolo(data_yaml, num_epochs, gpu_memory_limit, batch_size, img_size):
    # Initialize YOLO model
    model = YOLO("yolov5s.yaml")

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        model.to(device)
        torch.cuda.set_per_process_memory_fraction(gpu_memory_limit)

    # Set the batch size and image size
    model.batch_size = batch_size
    model.imgsz = img_size

    # Set the number of workers (nw) to 1 to reduce RAM usage
    nw = min([os.cpu_count(), 1])  # number of workers
    model.nc = 2  # Override the number of classes

    # Train the model with the updated settings
    results = model.train(data=data_yaml, epochs=num_epochs, batch=batch_size, imgsz=img_size, workers=nw)

    return results

if __name__ == "__main__":
    data_yaml = r"C:\Users\harli\Downloads\parasite detection.v1i.yolov8\data.yaml"
    num_epochs = 500
    gpu_memory_limit = 0.5  # Allocate approximately 2GB of GPU memory
    batch_size = 8
    img_size = 416  # Adjust image size as needed

    train_yolo(data_yaml, num_epochs, gpu_memory_limit, batch_size, img_size)
