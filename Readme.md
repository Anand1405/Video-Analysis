# Frame Selection and Object Detection in Videos

This repository provides an end-to-end pipeline for frame selection and shoppable object detection from videos. The implemented methods leverage deep learning models like CLIP, VGG16, and Grounding DINO, as well as traditional techniques such as histogram-based sampling and clustering.

---

## Features

### Frame Selection Methods
1. **Histogram Sampling**
   - Computes color histograms for each frame.
   - Selects diverse frames by maximizing pairwise histogram distances.

2. **Deep Learning-Based Selection**
   - Extracts features using the VGG16 model.
   - Performs clustering on deep features to select representative frames.

3. **Clustering-Based Selection**
   - Flattens pixel-level data of frames.
   - Clusters frames using KMeans to ensure diversity.

4. **CLIP-based Sampling**
   - Uses the CLIP model to compute embeddings for frames.
   - Scores frames based on similarity to a provided textual query.

### Object Detection
- Utilizes **Grounding DINO** for robust object detection.
- Supports shoppable item detection through text-prompted object detection.

### Key Features
- Multi-method comparison for frame selection.
- Integration of pre-trained models from PyTorch, TensorFlow, and Hugging Face.
- Results saved in JSON format for further analysis.
- Frame outputs stored in organized directories.

---

## Setup and Requirements

### Prerequisites
- Python 3.8+
- GPU with CUDA support (optional but recommended for faster processing)

### Install Dependencies
```bash
pip install -r requirements.txt
```
- Make sure you are in the main directory of the project.
```bash
streamlit run app.py
```

---

## Results from Gemini 2.0 Flash Experimental Model

### Note: This is not available in the code as Gemini 2.0 is only free on AI Studio UI
Prompt: Detect shoppable items in a 60-second video while minimizing computational time by selecting around 20 frames out of 900 for object detection. Provide frames and detect all the shoppable items in the video provided below, with total time taken and total detections.

### Time Taken: 5 Seconds

### Output (Gemini):
Okay, here's the analysis of the video with selected frames for object detection, aiming to balance accuracy with computational efficiency:

Frames Selected:

I've chosen frames that show a variety of different objects in the video, while skipping frames with duplicates or no items, approximately one frame every three seconds:

0:00, 0:03, 0:06, 0:09, 0:12, 0:15, 0:18, 0:21, 0:24, 0:27, 0:30, 0:33, 0:36, 0:39, 0:42, 0:45, 0:48, 0:51, 0:54, 0:57

Object Detections:

Here's a summary of the shoppable items detected within the selected frames:

0:00: Floating moon lamp on a wooden base
0:03: Digital tape measure, eggplant
0:06: Self-stirring mug with lid
0:09: Self-stirring mug with lid
0:12: Hand warmer, in hands and pockets
0:15: Kneeling desk chair
0:18: Kneeling desk chair
0:21: Battery converter
0:24: Handheld Ink Printer
0:27: Handheld Ink Printer
0:30: Measuring square cup
0:33: Pizza Slider Tray
0:36: Storage travel pillow
0:39: Cable zipper
0:42: Cable zipper
0:45: Stanley cup snack bowl
0:48: Magnetic Travel Bottles
0:51: Pill Dispenser
0:54: Pill Dispenser
0:57: Mug stackers, mugs

Summary:

Total Detections: 21 unique objects were detected.

Notes:

This method focused on detecting the majority of visually distinct items while keeping the computational load low.

Some objects might be present in multiple frames, but were only counted as a unique detection.

The location of a shoppable item doesn't mean it's being shown to purchase it, just to demonstrate the product.

If you have any additional requests or need more detailed detections, let me know!

---

## Results Comparison

### Note: Detection Count is not of unique objects but overall detection in all the selected frames

| Method                          | Gemini                          | Histogram Sampling              | Deep Learning Based Sampling    | Clip Based Sampling             | Clustering Based Sampling       |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| **Frames Extracted**            | 20                              | 20                              | 20                              | 20                              | 20                              |
| **Total Detections**            | 21                              | 86                              | 114                             | 163                             | 103                             |
| **Time Taken (seconds)**        | 5                               | 25.76                           | 476.60                          | 53.14                           | 68.22                           |

---

## Future Plan

- Use large models for embedding generation to perform analysis.
- Use large zero shot object detection models.
- Explore other methods using LLM's.

---

## Citations

- https://ieeexplore.ieee.org/document/9093615
- https://pub.aimind.so/efficient-frame-extraction-for-video-object-annotation-366daba84556
- https://www.semanticscholar.org/paper/Best-Frame-Selection-in-a-Short-Video-Ren-Shen/e94166bce09ca810b12e4b7a4ea920444893ffcf
- https://www.nature.com/articles/s41598-024-75923-y
