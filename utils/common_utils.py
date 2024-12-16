import cv2
import json
import numpy as np
import torch
import time
import os
from PIL import Image

# Convert numpy arrays to lists for JSON serialization
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")

# Preprocess frames for CLIP
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# Grounding DINO-based object detection
def detect_objects_with_grounding_dino(grounding_model, grounding_processor, device, image, text_query):
    inputs = grounding_processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]  # (height, width)
    )
    return results

# Save frames to directory
def save_frames_to_directory(frames, method_name):
    output_dir = f"frames_output/{method_name}"
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)