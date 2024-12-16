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
