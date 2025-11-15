import streamlit as st
import torch
import numpy as np
import librosa
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch.nn.functional as F
import plotly.graph_objects as go
import io
import os
import tempfile
from pathlib import Path

# ==========================================
# 1. CONFIG & SETUP
# ==========================================
class Config:
    BASE_DIR = Path(__file__).resolve().parent

    # Corrected paths to point to the 'models' subdirectory
    LOCAL_FER_PATH = BASE_DIR / "models" / "fer_model_final"
    LOCAL_SER_PATH = BASE_DIR / "models" / "ser_model_final"

    # Original model IDs for loading pre-processors
    FER_MODEL_ID = "google/vit-base-patch16-224-in21k"
    SER_MODEL_ID = "facebook/wav2vec2-base"

    # Tuned parameters
    ALPHA = 0.5
    BETA = 0.5
    DISTRESS_THRESHOLD = 0.55

    LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    DISTRESS_INDICES = [0, 2, 5] # angry, fear, sad


# ==========================================
# 2. MODEL LOADER
# ==========================================
@st.cache_resource
def load_models():
    """Loads models and processors. Processors are loaded from the web,
    while fine-tuned models are loaded from local disk."""
    st.info("Loading models — this may take a few seconds...")

    # Check for local model folders
    if not Config.LOCAL_FER_PATH.exists():
        st.error(f"❌ FER model directory not found at: {Config.LOCAL_FER_PATH}")
        st.stop()

    if not Config.LOCAL_SER_PATH.exists():
        st.error(f"❌ SER model directory not found at: {Config.LOCAL_SER_PATH}")
        st.stop()

    # --- Load FER Model & Processor ---
    try:
        # Load processor from web
        fer_extractor = ViTImageProcessor.from_pretrained(Config.FER_MODEL_ID)
        # Load fine-tuned model from local disk
        fer_model = AutoModelForImageClassification.from_pretrained(Config.LOCAL_FER_PATH)
    except Exception as e:
        st.error(f"Failed to load FER model: {e}")
        st.stop()

    # --- Load SER Model & Processor ---
    try:
        # Load processor from web
        ser_extractor = AutoFeatureExtractor.from_pretrained(Config.SER_MODEL_ID)
        # Load fine-tuned model from local disk
        ser_model = AutoModelForAudioClassification.from_pretrained(Config.LOCAL_SER_PATH)
    except Exception as e:
        st.error(f"Failed to load SER model: {e}")
        st.stop()

    st.success("✅ Models loaded successfully.")
    return fer_model, fer_extractor, ser_model, ser_extractor


# ==========================================
# 3. INFERENCE ENGINE (FIXED AUDIO LOADING)
# ==========================================
def predict_emotion(image=None, audio_file=None, models=None):
    fer_model, fer_ext, ser_model, ser_ext = models
    device = next(fer_model.parameters()).device # Get model's device (cpu or cuda)

    prob_f = np.zeros(len(Config.LABELS))
    w_f = 0.1 # Default low weight

    if image is not None:
        try:
            # Preprocess
            inputs = fer_ext(images=image, return_tensors="pt").to(device)
            # Run inference
            with torch.no_grad():
                logits = fer_model(**inputs).logits
            prob_f = F.softmax(logits, dim=-1).cpu().numpy()[0]

            # Calculate signal quality (adaptive weight)
            img_cv = np.array(image.convert('L'))
            w_f = np.clip((np.mean(img_cv) / 255.0) * (np.std(img_cv) / 255.0) * 10.0, 0.2, 0.9)
        except Exception as e:
            st.warning(f"Could not process image: {e}")
            pass

    prob_s = np.zeros(len(Config.LABELS))
    w_s = 0.1 # Default low weight

    if audio_file:
        temp_path = None
        try:
            # Get the original file extension (e.g., ".mp3")
            file_extension = os.path.splitext(audio_file.name)[1]
            
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(audio_file.read())
                temp_path = tmp.name # Get the full path to the temp file
            
            # Load audio from the temp file path
            y, sr = librosa.load(temp_path, sr=16000)
            
            if len(y) > 0:
                # Preprocess
                inputs = ser_ext(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
                # Run inference
                with torch.no_grad():
                    logits = ser_model(**inputs).logits
                prob_s = F.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Calculate signal quality (adaptive weight)
                w_s = np.clip(np.mean(y ** 2) * 1000, 0.2, 0.9)
        except Exception as e:
            st.warning(f"Could not process audio: {e}")
            pass
        finally:
            # Ensure the temp file is deleted
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


    # Normalize weights
    total_w = max(w_f + w_s, 1e-6) # Avoid division by zero
    w_f, w_s = w_f / total_w, w_s / total_w

    return prob_f, prob_s, w_f, w_s


# ==========================================
# 4. FUSION & DECISION LOGIC
# ==========================================
def fuse_and_decide(prob_f, prob_s, w_f, w_s):
    # E = argmax( w_f * P_f + w_s * P_s )
    fused_probs = (w_f * prob_f) + (w_s * prob_s)
    final_idx = int(np.argmax(fused_probs))
    final_emotion = Config.LABELS[final_idx]

    # D = alpha * P_f(distress) + beta * P_s(distress)
    p_distress_face = np.sum(prob_f[Config.DISTRESS_INDICES])
    p_stress_voice = np.sum(prob_s[Config.DISTRESS_INDICES])
    
    d_score = (Config.ALPHA * p_distress_face) + (Config.BETA * p_stress_voice)
    is_distress = d_score >= Config.DISTRESS_THRESHOLD

    return final_emotion, fused_probs, is_distress, d_score


# ==========================================
# 5. IoT SIMULATOR LOGIC
# ==========================================
def update_iot_simulator(emotion, is_distress):
    """Updates the st.session_state based on the detected emotion."""
    
    if is_distress:
        st.session_state.iot_devices['light']['state'] = "ALERT"
        st.session_state.iot_devices['light']['color'] = "#F44336" # Red
        st.session_state.iot_devices['thermostat']['state'] = "20°C (Stable)"
        st.session_state.iot_devices['music']['state'] = "Playing: Calming Tones"
        st.session_state.iot_devices['security']['state'] = "LOCKED"
    
    elif emotion == 'sad':
        st.session_state.iot_devices['light']['state'] = "Dimmed"
        st.session_state.iot_devices['light']['color'] = "#2196F3" # Soft Blue
        st.session_state.iot_devices['thermostat']['state'] = "22°C (Warmer)"
        st.session_state.iot_devices['music']['state'] = "Playing: Soothing Mix"
        st.session_state.iot_devices['security']['state'] = "Unlocked"

    elif emotion == 'angry':
        st.session_state.iot_devices['light']['state'] = "Bright"
        st.session_state.iot_devices['light']['color'] = "#B2EBF2" # Cool Blue
        st.session_state.iot_devices['thermostat']['state'] = "19°C (Cooler)"
        st.session_state.iot_devices['music']['state'] = "Playing: White Noise"
        st.session_state.iot_devices['security']['state'] = "Unlocked"
        
    elif emotion == 'happy':
        st.session_state.iot_devices['light']['state'] = "Bright"
        st.session_state.iot_devices['light']['color'] = "#FFF176" # Warm Yellow
        st.session_state.iot_devices['thermostat']['state'] = "21°C"
        st.session_state.iot_devices['music']['state'] = "Playing: Upbeat Music"
        st.session_state.iot_devices['security']['state'] = "Unlocked"
        
    else: # neutral, disgust, surprise
        st.session_state.iot_devices['light']['state'] = "On"
        st.session_state.iot_devices['light']['color'] = "#FFFFFF" # Neutral White
        st.session_state.iot_devices['thermostat']['state'] = "21°C"
        st.session_state.iot_devices['music']['state'] = "Off"
        st.session_state.iot_devices['security']['state'] = "Unlocked"

# ==========================================
# 6. STREAMLIT UI (REBUILT FOR ROBUSTNESS)
# ==========================================
def main():
    st.set_page_config(page_title="Cognitive Home Guardian", page_icon="🛡️", layout="wide")
    
    # --- CSS Styling ---
    st.markdown("""
    <style>
        .stButton>button { 
            height: 3em; 
            width: 100%; 
            background-color: #4CAF50; 
            color: white; 
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .metric-box { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #2196F3; 
            margin-bottom: 10px;
        }
        .alert-box { 
            background-color: #ffebee; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #f44336; 
            margin-bottom: 10px;
        }
        .stMetric {
            border-radius: 10px;
            padding: 10px;
            background-color: #f8f9fa;
        }
        /* Style for the IoT device color dot */
        .color-dot {
            width: 25px;
            height: 25px;
            border-radius: 50%;
            border: 1px solid #ddd;
            display: inline-block;
            vertical-align: middle;
            margin-left: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Cognitive Home Guardian (with IoT Simulator)")
    st.caption("Real-Time Emotion Aware IoT System | Running Locally")

    # --- Initialize Session State for IoT Devices ---
    if 'iot_devices' not in st.session_state:
        st.session_state.iot_devices = {
            "light": {"name": "Living Room Light", "icon": "💡", "state": "On", "color": "#FFFFFF"},
            "thermostat": {"name": "Thermostat", "icon": "🌡️", "state": "21°C"},
            "music": {"name": "Music Player", "icon": "🎵", "state": "Off"},
            "security": {"name": "Front Door", "icon": "🔒", "state": "Unlocked"}
        }

    # --- Load Models ---
    models = load_models()
    
    # --- UI Layout ---
    col1, col2, col3 = st.columns([1.2, 1.8, 1])
    
    # --- COLUMN 1: SENSORS ---
    with col1:
        st.subheader("1. Real-World Sensing")
        img_file = st.camera_input("Capture Face")
        audio_file = st.file_uploader("Record/Upload Voice Clip", type=['wav', 'mp3', 'm4a'])
        analyze_btn = st.button("🧠 Analyze Environment")

    # --- COLUMN 2: AI ANALYSIS ---
    with col2:
        st.subheader("2. Cognitive Decision")

        if analyze_btn:
            if not img_file and not audio_file:
                st.warning("Please provide at least one input (Face or Voice).")
            else:
                with st.spinner("Processing Local Inference..."):
                    image = Image.open(img_file) if img_file else None
                    
                    p_f, p_s, w_f, w_s = predict_emotion(image, audio_file, models)
                    emotion, probs, distress, d_score = fuse_and_decide(p_f, p_s, w_f, w_s)
                    
                    # Update the IoT simulator state
                    update_iot_simulator(emotion, distress)
                    
                    # --- This section now draws directly ---
                    st.markdown("<b>Adaptive Weights:</b>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Vision Weight (w_f)", f"{w_f:.2f}")
                    c2.metric("Voice Weight (w_s)", f"{w_s:.2f}")
                    
                    if distress:
                        st.markdown(f"""
                        <div class="alert-box">
                            <h3>🚨 DISTRESS DETECTED (Score: {d_score:.2f})</h3>
                            <b>System Actions Triggered. See IoT Dashboard.</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h3>State: {emotion.upper()}</h3>
                            <b>IoT Response:</b> Adjusting ambient settings...
                        </div>
                        """, unsafe_allow_html=True)
                
                    # Visualization
                    fig = go.Figure(data=[
                        go.Bar(name='Vision', x=Config.LABELS, y=p_f * w_f, marker_color='#90CAF9'),
                        go.Bar(name='Voice', x=Config.LABELS, y=p_s * w_s, marker_color='#FFCC80'),
                        go.Scatter(name='Fused', x=Config.LABELS, y=probs, mode='lines+markers', line=dict(color='green', width=3))
                    ])
                    fig.update_layout(title="Probabilistic Fusion (E)", barmode='stack', height=300, margin=dict(l=0,r=0,b=0,t=40))
                    st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Click 'Analyze Environment' to see the AI decision.")

    # --- COLUMN 3: IOT SIMULATOR DASHBOARD (REBUILT) ---
    with col3:
        st.subheader("🏠 Virtual Home")
        
        # Get the current state
        iot_state = st.session_state.iot_devices
        
        # --- Light Control (with color) ---
        with st.container(border=True):
            st.markdown(f"<b>{iot_state['light']['icon']} {iot_state['light']['name']}</b>", unsafe_allow_html=True)
            light_col1, light_col2 = st.columns([3, 1])
            with light_col1:
                st.markdown(f"<p style='margin: 0; padding: 0;'>{iot_state['light']['state']}</p>", unsafe_allow_html=True)
            with light_col2:
                st.markdown(f"<div class='color-dot' style='background-color: {iot_state['light']['color']};'></div>", unsafe_allow_html=True)
        
        # --- Other Devices (using st.metric) ---
        with st.container(border=True):
            st.metric(
                label=f"{iot_state['thermostat']['icon']} {iot_state['thermostat']['name']}",
                value=iot_state['thermostat']['state']
            )
            
        with st.container(border=True):
            st.metric(
                label=f"{iot_state['music']['icon']} {iot_state['music']['name']}",
                value=iot_state['music']['state']
            )

        with st.container(border=True):
            st.metric(
                label=f"{iot_state['security']['icon']} {iot_state['security']['name']}",
                value=iot_state['security']['state']
            )

if __name__ == "__main__":
    main()