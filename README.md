# ASL Sign Language Interpreter

A real-time American Sign Language (ASL) interpreter using deep learning and computer vision. This application uses an LSTM neural network trained on the Sign Language MNIST dataset to recognize ASL letters A-Y in real-time through your webcam.

## ✨ Features

- **Real-time Recognition**: Live webcam feed with instant ASL letter recognition
- **Hand Detection**: Uses MediaPipe for robust hand landmark detection
- **LSTM Model**: Deep learning model trained on Sign Language MNIST dataset
- **Interactive UI**: Clean Streamlit interface with easy controls
- **High Accuracy**: Preprocessed input matching training data format

## 🎯 Supported Signs

The model recognizes ASL letters **A through Y** (excluding J, as it requires motion and is not included in the static Sign Language MNIST dataset).

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam/Camera access
- Minimum 4GB RAM recommended
- Trained model files

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd SignLanguageInterpreter
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Model Files
Ensure you have the following files in the `model/` directory:
- `asl_lstm_model.h5` - Trained LSTM model
- `label_map.json` - Label mapping for predictions

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Access the App
After you run locally, open your browser and navigate to `http://localhost:8501`


### To Train Your Own Model
```bash
# Ensure training data is in data/ directory
jupyter notebook train.ipynb
# Follow the notebook instructions
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Hand detection and landmark extraction
- **TensorFlow**: Deep learning framework for LSTM model
- **NumPy**: Numerical computations