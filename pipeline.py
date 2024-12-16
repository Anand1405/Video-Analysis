import cv2
import json
import numpy as np
import torch
import time
import os
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotObjectDetection
from keras.applications import VGG16
from utils.common_utils import save_frames_to_directory, preprocess_frame, numpy_to_list, detect_objects_with_grounding_dino
from utils.frame_extract import select_frames_histogram, select_frames_deep_learning, select_frames_clustering, select_frames_clip

# Select device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP model for image embeddings
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

# Load the Grounding DINO model
grounding_model_id = "IDEA-Research/grounding-dino-base"
grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet", include_top=False)

# Pipeline
def pipe(video_path, num_frames, method_name, query_text):
    methods = {
        "Histogram Sampling": select_frames_histogram,
        "Deep Learning-Based": select_frames_deep_learning,
        "Clustering-Based": select_frames_clustering,
        "CLIP-based Sampling": select_frames_clip,
    }

    results = {}
    try:
        font = ImageFont.truetype("arial.ttf", size=20)  # Change to a valid font file path if needed
    except IOError:
        font = ImageFont.load_default()
        
    for method_name, method in methods.items():
        start_time = time.time()
        if method_name == "CLIP-based Sampling":
            selected_frames = method(clip_model, clip_processor, video_path, num_frames, query_text)
        elif method_name == "Deep Learning-Based":
            selected_frames = method(vgg_model, video_path, num_frames)
        else:
            selected_frames = method(video_path, num_frames)

        save_frames_to_directory(selected_frames, method_name)

        detections = []
        for i, frame in enumerate(selected_frames):
            image = preprocess_frame(frame)
            draw = ImageDraw.Draw(image)
            result = detect_objects_with_grounding_dino(grounding_model, grounding_processor, device, image, query_text)
            for i in range(len(result[0]['boxes'])):
                box = result[0]['boxes'][i].cpu().numpy().astype(int)  # Convert box to numpy array and to int
                # Draw the bounding box
                draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=3)
                
                # Put the label above the bounding box
                draw.text((box[0], box[1] - 10), query_text, fill="white", font=font)
            selected_frames[i] = image
            detections.append(result)

        elapsed_time = time.time() - start_time
        results[method_name] = {
            "time_taken": elapsed_time,
            "detections": len(detections),
            "detection_boxes": [detection for detection in detections],
        }

    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, default=numpy_to_list)
        
    return results, selected_frames
