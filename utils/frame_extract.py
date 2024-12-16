import numpy as np
import cv2
import torch
import os
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from keras.applications.vgg16 import preprocess_input
from utils.common_utils import preprocess_frame

# Compute CLIP image embedding
def compute_clip_embedding(clip_model, clip_processor, device, frame):
    image = preprocess_frame(frame)
    inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze(0).cpu().numpy()

# Compute text embedding using CLIP
def compute_text_embedding(clip_model, clip_processor, device, text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    return embedding.squeeze(0).cpu().numpy()

# Compute histogram for frame
def compute_histogram(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Compute deep features using VGG16
def compute_deep_features(vgg_model, frame):
    frame_preprocessed = preprocess_input(np.expand_dims(frame, axis=0))
    features = vgg_model.predict(frame_preprocessed)
    return features.flatten()

# Frame selection methods

# Select frames using Historgram Sampling
def select_frames_histogram(frames, num_frames=20):
    histograms = []
    for frame in frames:
        histograms.append(compute_histogram(frame))

    dist_matrix = cdist(histograms, histograms, metric="euclidean")
    selected_indices = []
    selected_frames = []

    for i in range(num_frames):
        if not selected_indices:
            selected_indices.append(i)
            selected_frames.append(frames[i])
        else:
            max_dist = -1
            idx_to_add = None
            for j in range(len(frames)):
                if j not in selected_indices:
                    dist = np.min(dist_matrix[j, selected_indices])
                    if dist > max_dist:
                        max_dist = dist
                        idx_to_add = j
            selected_indices.append(idx_to_add)
            selected_frames.append(frames[idx_to_add])

    return selected_frames

# Select frames using VGG16 and clustering
def select_frames_deep_learning(vgg_model, frames, num_frames=20):
    features = []

    for frame in frames:
        features.append(compute_deep_features(vgg_model, frame))

    # Check if features are empty
    if not features:
        print("No frames were processed. Please check the video path.")
        return []

    # Convert features to a numpy array
    features = np.array(features)

    # Check if features have the correct shape
    if features.ndim != 2:
        print("Features array is not 2D. Please check feature extraction.")
        return []

    kmeans = KMeans(n_clusters=num_frames, n_init='auto')
    cluster_indices = kmeans.fit_predict(features)

    # Select one frame per cluster
    selected_frames_dict = {}
    for i in range(len(cluster_indices)):
        cluster_id = cluster_indices[i]
        if cluster_id not in selected_frames_dict:
            selected_frames_dict[cluster_id] = frames[i]

    selected_frames = list(selected_frames_dict.values())

    # Ensure we only return up to num_frames
    if len(selected_frames) > num_frames:
        selected_frames = selected_frames[:num_frames]

    return selected_frames


# Select frames using KMeans Clustering
def select_frames_clustering(frames, num_frames=20):
    flat_frames = [frame.flatten() for frame in frames]
    kmeans = KMeans(n_clusters=num_frames, n_init='auto')
    cluster_indices = kmeans.fit_predict(flat_frames)
    # Create a dictionary to hold selected frames for each cluster
    selected_frames_dict = {}

    # Loop through the cluster indices and select one frame per cluster
    for i in range(len(cluster_indices)):
        cluster_id = cluster_indices[i]
        if cluster_id not in selected_frames_dict:
            selected_frames_dict[cluster_id] = frames[i].reshape(256, 256, -1)

    # Convert the dictionary values to a list to get the selected frames
    selected_frames = list(selected_frames_dict.values())

    # If you want to ensure that you only have 'num_clusters' frames:
    num_clusters = len(selected_frames_dict)
    if num_clusters > num_frames:
        selected_frames = selected_frames[:num_frames]

    return selected_frames

# Select frames using Clip Model
def select_frames_clip(clip_model, clip_processor, device, frames, num_frames=20, query_text="shoppable item"):
    text_embedding = compute_text_embedding(clip_model, clip_processor, device, query_text)
    embeddings = []

    # Extract frames and their embeddings
    for frame in frames:
        embeddings.append(compute_clip_embedding(clip_model, clip_processor, device, frame))

    if len(frames) == 0:
        raise ValueError(f"No frames captured from the video. Please check the video file.")

    if len(frames) <= num_frames:
        # If fewer frames than needed, return all frames
        return frames + [frames[-1]] * (num_frames - len(frames))

    # Compute similarity scores with text embedding
    similarities = np.array([
        np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb))
        for emb in embeddings
    ])
    
    # Sort frames by similarity
    sorted_indices = similarities.argsort()[::-1]

    # Select top candidates for diversity selection
    top_candidates = [embeddings[i] for i in sorted_indices[:3 * num_frames]]
    candidate_frames = [frames[i] for i in sorted_indices[:3 * num_frames]]

    # Greedy selection for diversity
    selected_indices = [0]  # Start with the most similar frame
    for _ in range(1, num_frames):
        max_diversity_idx = None
        max_diversity_score = -1

        for idx, emb in enumerate(top_candidates):
            if idx in selected_indices:
                continue

            # Calculate diversity score as the minimum distance to selected frames
            diversity_score = min(
                np.linalg.norm(emb - top_candidates[selected_idx]) for selected_idx in selected_indices
            )

            if diversity_score > max_diversity_score:
                max_diversity_score = diversity_score
                max_diversity_idx = idx

        selected_indices.append(max_diversity_idx)

    # Retrieve final selected frames
    selected_frames = [candidate_frames[idx] for idx in selected_indices]

    return selected_frames
