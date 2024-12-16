import streamlit as st
import os
import cv2
from PIL import Image
from pipeline import pipe
import warnings
warnings.filterwarnings("ignore")

# Streamlit UI
st.title("Video Frame Sampling Tool")
st.sidebar.title("Options")

# Upload video file
uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi"])
method_name = st.sidebar.selectbox(
    "Select Sampling Method",
    [
        "Histogram Sampling",
        "Deep Learning-Based Sampling",
        "CLIP-based Sampling",
        "Clustering Sampling",
    ],
)

num_frames = st.sidebar.number_input(
    "Number of Frames to Extract", 
    min_value=1, 
    max_value=100, 
    value=20, 
    step=1
)

query_text = st.sidebar.text_input("Query Text for CLIP-based Sampling", "shoppable item.")
if not query_text.endswith("."):
    query_text = query_text + "."

if uploaded_file:
    # Save uploaded file temporarily
    video_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if st.button("Process Video"):
        st.write(f"Processing with {method_name}...")
        results, selected_frames = pipe(video_path, num_frames, method_name, query_text)
        st.write(f"{len(selected_frames)} frames extracted.")
        if len(selected_frames) > 0:
            st.write(f"Total Detections: {results[method_name]['detections']}")
            st.write(f"Time Taken: {results[method_name]['time_taken']}")
        
        # Display saved frames
        st.write("Extracted Frames:")
        for idx, frame in enumerate(selected_frames):
            st.image(frame, caption=f"Frame {idx + 1}")
            
        
