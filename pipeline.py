import cv2
import json
import numpy as np
import torch
import time
import os
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotObjectDetection
from keras.applications import VGG16
from utils.common_utils import get_frames, save_frames_to_directory, preprocess_frame, numpy_to_list, detect_objects_with_grounding_dino
from utils.frame_extract import select_frames_histogram, select_frames_deep_learning, select_frames_clustering, select_frames_clip

# Select device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load the CLIP model for image embeddings
clip_processor = CLIPProcessor.from_pretrained("models/CLIP-Model/processor")
clip_model = CLIPModel.from_pretrained("models/CLIP-Model/model").to(device)

# Load the Grounding DINO model
grounding_processor = AutoProcessor.from_pretrained("models/Grounding-Dino-Model/processor")
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("models/Grounding-Dino-Model/model").to(device)

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet", include_top=False, pooling="avg", input_shape=(256, 256, 3))

print("All models loaded successfully!!!")

# Pipeline
def pipe(video_path, num_frames, method_name, query_text):
    methods = {
        "Histogram Sampling": select_frames_histogram,
        "Deep Learning-Based Sampling": select_frames_deep_learning,
        "Clustering Sampling": select_frames_clustering,
        "CLIP-based Sampling": select_frames_clip,
    }

    results = {}
    try:
        font = ImageFont.truetype("arial.ttf", size=20)  # Change to a valid font file path if needed
    except IOError:
        font = ImageFont.load_default()
        
    print("Video Path:", video_path)
    frames = get_frames(video_path)
        
    start_time = time.time()
    if method_name == "CLIP-based Sampling":
        selected_frames = methods[method_name](clip_model, clip_processor, device, frames, num_frames, query_text)
    elif method_name == "Deep Learning-Based Sampling":
        selected_frames = methods[method_name](vgg_model, frames, num_frames)
    else:
        selected_frames = methods[method_name](frames, num_frames)
            
    print("Frames Selected!!!")
        
    save_frames_to_directory(selected_frames, method_name)
    print(f"{len(selected_frames)} Selected Frames Saved!!!")
        
    total = 0
    detections = []
    for curr in range(len(selected_frames)):
        image = preprocess_frame(selected_frames[curr])
            
        print(f"Starting Object Detection: {curr}----------------")
        result = detect_objects_with_grounding_dino(grounding_model, grounding_processor, device, image, query_text)
        print("Object Detection Complete----------------")
            
        draw = ImageDraw.Draw(image)
        print("Creating Bounding Box---------------")
        for currBox in range(len(result[0]['boxes'])):
            total += 1
            box = result[0]['boxes'][currBox].cpu().numpy().astype(int)  # Convert box to numpy array and to int
            # Draw the bounding box
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=3)
                
            # Put the label above the bounding box
            draw.text((box[0], box[1] - 10), query_text, fill="white", font=font)
        selected_frames[curr] = image
        print("Bounding Box Created----------------")
        detections.append(result)

        
        print(total)
        elapsed_time = time.time() - start_time
        results[method_name] = {
            "time_taken": elapsed_time,
            "detections": total,
            "detection_boxes": [detection for detection in detections],
        }

    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, default=numpy_to_list)
        
    return results, selected_frames
