import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
import time

# Configuration
MODEL_PATH = "model/asl_lstm_model.h5"
LABEL_MAP_PATH = "model/label_map.json"

st.set_page_config(page_title="ASL Camera Interpreter", layout="wide")
st.title("ASL Sign Language Interpreter")

@st.cache_resource
def load_model_and_map():
    """Load the trained model and label mapping"""
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        return model, label_map
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_hand_region(hand_roi):
    """Preprocess hand region to match training data format"""
    # Convert to grayscale if needed
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    # Check if background is dark and invert if needed
    if normalized.mean() < 0.4:
        normalized = 1.0 - normalized
    
    # Reshape for model input
    return normalized.reshape(1, 28, 28)

def capture_and_predict():
    """Capture image and make prediction"""
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access camera. Please check permissions.")
        return
    
    # Model loading
    model, label_map = load_model_and_map()
    if model is None:
        st.error("❌ Model not found. Please train the model first.")
        cap.release()
        return
    
    class_to_letter = label_map.get("class_to_letter", {})
    
    # Streamlit layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        image_placeholder = st.empty()
        # Add stop button
        if st.button("🛑 Stop Camera", type="secondary", key="stop_btn"):
            st.session_state.stop_camera = True
    
    with col2:
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.markdown("### Instructions:")
        st.markdown("- Position your hand in the camera view")
        st.markdown("- Make clear ASL letter signs")
        st.markdown("- Keep good lighting")
        st.markdown("- Letters A-Y supported (no J)")
        st.markdown("- Click **Stop Camera** to stop")
        
        if st.button("📸 Capture & Predict", type="primary"):
            st.session_state.capture_now = True
    
    # Initialize hands detector
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        frame_count = 0
        last_prediction = None
        last_confidence = 0.0
        
        # Main camera loop
        while True:
            # Check for stop signal
            if st.session_state.get('stop_camera', False):
                st.session_state.stop_camera = False
                st.success("📹 Camera stopped!")
                break
                
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks and bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get bounding box
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords) - 20), int(max(x_coords) + 20)
                    y_min, y_max = int(min(y_coords) - 20), int(max(y_coords) + 20)
                    
                    # Ensure bounds are within frame
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Extract and predict every 10 frames or on capture button
                    if frame_count % 10 == 0 or st.session_state.get('capture_now', False):
                        if st.session_state.get('capture_now', False):
                            st.session_state.capture_now = False
                        
                        # Extract hand region
                        hand_roi = frame[y_min:y_max, x_min:x_max]
                        
                        if hand_roi.size > 0:
                            try:
                                # Preprocess
                                processed_hand = preprocess_hand_region(hand_roi)
                                
                                # Predict
                                predictions = model.predict(processed_hand, verbose=0)
                                predicted_class = np.argmax(predictions[0])
                                confidence = predictions[0][predicted_class]
                                letter = class_to_letter.get(str(predicted_class), str(predicted_class))
                                
                                last_prediction = letter
                                last_confidence = confidence
                                
                            except Exception as e:
                                last_prediction = "Error"
                                last_confidence = 0.0
                    
                    # Display current prediction on frame
                    if last_prediction:
                        cv2.putText(frame, f"Letter: {last_prediction}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {last_confidence*100:.1f}%", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            else:
                cv2.putText(frame, "Show your hand to the camera", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert back to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update Streamlit display
            with image_placeholder:
                st.image(frame_rgb, caption="Camera Feed", use_container_width=True)
            
            if last_prediction:
                with prediction_placeholder:
                    st.metric("Predicted Letter", last_prediction)
                with confidence_placeholder:
                    st.metric("Confidence", f"{last_confidence*100:.1f}%")
            
            frame_count += 1
            
            # Check if we should stop (this is a simplified check)
            # In a real app, you'd want a proper stop mechanism
            time.sleep(0.1)
    
    cap.release()

# Initialize session state
if 'stop_camera' not in st.session_state:
    st.session_state.stop_camera = False

# Run the app
if __name__ == "__main__":
    if st.button("🎥 Start Camera", type="primary"):
        capture_and_predict()