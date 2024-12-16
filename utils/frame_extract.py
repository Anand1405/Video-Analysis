import numpy as np
import cv2
import torch
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from keras.applications.vgg16 import preprocess_input
from utils.common_utils import preprocess_frame

# Compute CLIP image embedding
def compute_clip_embedding(clip_model, clip_processor, frame):
    image = preprocess_frame(frame)
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze(0).cpu().numpy()

# Compute text embedding using CLIP
def compute_text_embedding(clip_model, clip_processor, text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
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
    frame_resized = cv2.resize(frame, (224, 224))
    frame_preprocessed = preprocess_input(np.expand_dims(frame_resized, axis=0))
    features = vgg_model.predict(frame_preprocessed)
    return features.flatten()

# Frame selection methods

# Select frames using Historgram Sampling
def select_frames_histogram(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    histograms = []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        histograms.append(compute_histogram(frame))
        frames.append(frame)

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

    cap.release()
    return selected_frames

# Select frames using VGG16 and clustering
def select_frames_deep_learning(vgg_model, video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    features = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (256, 256))
        features.append(compute_deep_features(vgg_model, frame_resized))
        frames.append(frame_resized)

    kmeans = KMeans(n_clusters=num_frames, n_init='auto')
    cluster_indices = kmeans.fit_predict(features)
    selected_frames = [frames[i] for i in range(len(frames)) if cluster_indices[i] == cluster_indices[0]]

    cap.release()
    return selected_frames

# Select frames using KMeans Clustering
def select_frames_clustering(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (256, 256))
        frames.append(frame_resized.flatten())

    kmeans = KMeans(n_clusters=num_frames, n_init='auto')
    cluster_indices = kmeans.fit_predict(frames)
    selected_frames = [frames[i].reshape(256, 256, -1) for i in range(len(frames)) if cluster_indices[i] == cluster_indices[0]]

    cap.release()
    return selected_frames

# Select frames using Clip Model
def select_frames_clip(clip_model, clip_processor, video_path, num_frames=20, query_text="shoppable item"):
    cap = cv2.VideoCapture(video_path)
    text_embedding = compute_text_embedding(query_text)
    embeddings = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        embeddings.append(compute_clip_embedding(frame))
        frames.append(frame)

    similarities = np.array([np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb)) for emb in embeddings])
    selected_indices = similarities.argsort()[-num_frames:][::-1]
    selected_frames = [frames[i] for i in selected_indices]

    cap.release()
    return selected_frames