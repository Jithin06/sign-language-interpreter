# # app.py
# import streamlit as st
# from PIL import Image, ImageOps
# import numpy as np
# import tensorflow as tf
# import json
# import os
# import io
# import cv2

# MODEL_PATH = "model/asl_lstm_model.h5"
# LABEL_MAP_PATH = "model/label_map.json"

# st.set_page_config(page_title="ASL LSTM Interpreter", layout="centered")

# st.title("Sign Language Interpreter (Image → Letter)")
# st.write("Upload a 28×28 or larger image of a single hand sign (ASL static alphabet). Model expects the Sign Language MNIST style signs (A–Y; J and Z are motion-based and typically not included).")

# # Load model & label map once
# @st.cache_resource
# def load_model_and_map():
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model file not found at {MODEL_PATH}. Please train the model first using train.py.")
#         return None, None
#     model = tf.keras.models.load_model(MODEL_PATH)
#     with open(LABEL_MAP_PATH, "r") as f:
#         label_map = json.load(f)
#     return model, label_map

# model, label_map = load_model_and_map()
# if model is None:
#     st.stop()

# st.sidebar.header("Options")
# show_example = st.sidebar.checkbox("Show random test example from dataset (if available)", value=False)

# uploaded_file = st.file_uploader("Upload an image (PNG/JPG). For best results, upload 28x28 grayscale images like in the dataset.", type=["png","jpg","jpeg"])

# def preprocess_image_pil(pil_img):
#     """
#     Convert uploaded image to 28x28 grayscale, normalize and return shape (1,28,28)
#     """
#     # convert RGBA or RGB to grayscale
#     img = pil_img.convert("L")  # grayscale
#     # Option: crop to center square first
#     w, h = img.size
#     min_dim = min(w, h)
#     left = (w - min_dim) // 2
#     top = (h - min_dim) // 2
#     img = img.crop((left, top, left + min_dim, top + min_dim))
#     img = img.resize((28, 28), Image.ANTIALIAS)
#     arr = np.array(img).astype(np.float32)
#     arr = arr / 255.0
#     # In dataset, background may be white and hand dark. If your uploaded image is inverted, predictions will fail.
#     # Optionally invert if background dark (simple heuristic)
#     # If mean pixel is >0.6 assume background is white (dataset-like), else invert if necessary — keep heuristic simple
#     if arr.mean() < 0.4:
#         # background dark, invert to match dataset where background is light
#         arr = 1.0 - arr
#     # reshape to (1, 28, 28)
#     return arr.reshape(1, 28, 28)

# def predict_from_array(arr):
#     preds = model.predict(arr)
#     idx = np.argmax(preds, axis=1)[0]
#     prob = preds[0, idx]
#     # decode label
#     # class_to_letter mapping maps dataset numeric class to a letter
#     c2l = label_map.get("class_to_letter", {})
#     # if label_map uses class numbers as strings, convert:
#     letter = c2l.get(str(idx), c2l.get(int(idx), str(idx)))
#     return idx, letter, float(prob)

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded image", width=240)
#     arr = preprocess_image_pil(image)
#     idx, letter, prob = predict_from_array(arr)
#     st.markdown(f"**Prediction:** {letter} (class index {idx})")
#     st.markdown(f"**Confidence:** {prob*100:.2f}%")

#     st.write("You can try a few different images or adjust lighting/background to match dataset style (light background, hand sign centered).")

# # Optional: show an example image from the CSV dataset
# if show_example:
#     import pandas as pd
#     csv_path = os.path.join("data", "sign_mnist_test.csv")
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path)
#         sample = df.sample(1).iloc[0]
#         label = int(sample['label'])
#         pixels = sample.drop('label').values.reshape(28,28).astype(np.uint8)
#         st.write(f"Example from dataset — class {label}, mapped to letter: {label_map.get('class_to_letter',{}).get(str(label), label)}")
#         st.image(pixels, width=200, clamp=True, channels="L")
#     else:
#         st.info("No dataset CSV found in data/. Put sign_mnist_test.csv into data/ if you want to preview examples.")


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json

# Load model and label map
MODEL_PATH = "model/asl_lstm_model.h5"
LABEL_MAP_PATH = "model/label_map.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
class_to_letter = label_map.get("class_to_letter", {})

mp_hands = mp.solutions.hands

class HandSignTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        if result.multi_hand_landmarks:
            # take first hand
            landmarks = result.multi_hand_landmarks[0]
            # flatten into 63-d vector (x,y,z for 21 landmarks)
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            # reshape to model input shape: (1,63) or (1, timesteps, features) depending on your model
            input_data = landmark_array.reshape(1, -1)
            preds = model.predict(input_data)
            idx = int(np.argmax(preds))
            letter = class_to_letter.get(str(idx), str(idx))
            confidence = float(preds[0, idx])
            cv2.putText(img, f"{letter} ({confidence*100:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2, cv2.LINE_AA)
        return img

st.title("Real-Time Sign Language Interpreter (Webcam)")

webrtc_streamer(key="hand-sign", video_processor_factory=HandSignTransformer)
