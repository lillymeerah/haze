import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time

# ==========================
# 1. Configuration & Model Loading
# ==========================

# Filenames for the face detection model
FACE_PROTO = r"C:\Users\bibiz\Downloads\input data.txt"
FACE_MODEL = r"C:\Users\bibiz\Downloads\res10_300x300_ssd_iter_140000.caffemodel"

# Attempt to load the DNN face detector
if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
    face_net = None
    st.error(
        "Missing face detection model files. Please ensure both "
        "deploy.prototxt.txt and res10_300x300_ssd_iter_140000.caffemodel "
        "are present in the application directory."
    )
else:
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# ==========================
# 2. Face Blurring Function
# ==========================
def mask_faces_dnn(image_np, confidence_threshold=0.5):
    if face_net is None:
        raise RuntimeError("Face detection model not loaded.")

    (h, w) = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_np, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    blurred_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue

        box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            face_region = image_np[y1:y2, x1:x2]
            image_np[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (51, 51), 30)
            blurred_count += 1

    return image_np, blurred_count

# ==========================
# 3. Haze Detection (Dark Channel Prior)
# ==========================
def dark_channel(im, size=15):
    min_channel = np.min(im, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def get_haze_level(image_np):
    dark = dark_channel(image_np)
    haze_score = np.mean(dark) / 255.0  # Normalize
    if haze_score > 0.7:
        return "Clear"
    elif haze_score > 0.4:
        return "Medium Haze"
    else:
        return "Heavy Haze"

# ==========================
# 4. Streamlit App Interface
# ==========================
st.title("🌫️ Haze Detection with Face Blurring")

mode = st.radio("Input Source:", ["Upload Image", "Use Webcam"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        try:
            img_eff, face_count = mask_faces_dnn(img_np)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        haze_level = get_haze_level(img_eff)
        st.image(img_eff, caption=f"Faces blurred: {face_count}", use_column_width=True)
        st.success(f"Haze Level: **{haze_level}**")

elif mode == "Use Webcam":
    run_webcam = st.checkbox("Activate Webcam")
    frame_display = st.image([])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                frame_eff, face_count = mask_faces_dnn(frame)
            except RuntimeError as e:
                st.error(str(e))
                break

            haze_level = get_haze_level(frame_eff)
            frame_display.image(frame_eff, caption=f"Faces blurred: {face_count} | Haze Level: {haze_level}")
            time.sleep(0.05)

        cap.release()
