🛡️ Cognitive Home Guardian
===========================

This project is a software implementation of the "Cognitive Home Guardian" (CHG), a privacy-preserving, emotion-aware smart home system. It uses multimodal (face + voice) deep learning models to understand a user's emotional state and simulate an adaptive IoT response, all while running on-device to protect user privacy.

This repository contains the Streamlit web application that serves as the front-end and inference engine for the project.

_(Note: You will need to update this URL once your app is deployed)_

 Key Features
---------------

*   **Multimodal Emotion Fusion:** Implements the paper's core fusion logic (E = argmax(wf\*Pf + ws\*Ps)) by combining real-time inputs from a webcam and microphone.
    
*   **Adaptive Weights:** Automatically calculates sensor reliability (w\_f, w\_s) based on image brightness and audio volume.
    
*   **Distress Detection:** Uses the D = α\*Pf + β\*Ps formula to identify critical distress states (e.g., fear, sadness) and triggers a different response.
    
*   **Privacy First:** All AI inference runs locally in the browser or on the server. The models are loaded from Hugging Face, but your personal camera/mic data is never saved or uploaded.
    
*   **IoT Simulator:** A "Virtual Home" dashboard shows how real IoT devices (lights, thermostat, music) would react to your emotional state in real-time.
    

 Technology Stack
-------------------

*   **Application:** Streamlit
    
*   **AI / Deep Learning:** PyTorch
    
*   **Models:** Transformers (ViT & Wav2Vec2)
    
*   **Data Processing:** Librosa (Audio), OpenCV (Video)
    
*   **Visualization:** Plotly
    
*   **Model Hosting:** Hugging Face Hub
    
*   **Code Hosting:** GitHub
    
*   **App Deployment:** Streamlit Cloud
    

How to Deploy Your Own Version
---------------------------------

Follow these three steps to deploy this application.

### Step 1: Host Your Models on Hugging Face Hub

Your trained models (fer\_model\_final and ser\_model\_final) are too large for GitHub. They must be hosted for free on Hugging Face.

1.  **Create Account:** Go to [HuggingFace.co](https://huggingface.co/) and sign up.
    
2.  **New Repo:** Create a new **public Model repository**. Name it CognitiveHomeGuardian.
    
3.  **Upload Files:** Go to the "Files" tab of your new repo and click "Add file" -> "Upload folder".
    
4.  Upload your **fer\_model\_final** folder.
    
5.  Upload your **ser\_model\_final** folder.
    

When you are done, your repository structure should look like this:

`   Sankeerth28/CognitiveHomeGuardian/  ├── fer_model_final/  │   ├── config.json  │   └── model.safetensors  └── ser_model_final/      ├── config.json      └── model.safetensors   `

### Step 2: Push Your Code to GitHub

Your GitHub repository holds the application _code_.

1.  Make sure your project folder contains these four files:
    
    *   app.py (The main application code)
        
    *   requirements.txt (The list of libraries)
        
    *   .gitignore (To ignore local folders like .venv)
        
    *   README.md (This file)
        
2.  REPO\_ID = "Sankeerth28/CognitiveHomeGuardian"
    
3.  git add app.py requirements.txt .gitignore README.mdgit commit -m "Add readme and prepare for deployment"git push origin main
    

### Step 3: Deploy on Streamlit Cloud

1.  **Log In:** Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
    
2.  **New App:** Click **"New app"**.
    
3.  **Connect Repo:** Select your Sankeerth28/CognitiveHomeGuardian repository.
    
4.  **Branch:** main
    
5.  **Main file path:** app.py
    
6.  **Deploy!**
    

The app will take 2-3 minutes to install the libraries (from requirements.txt) and download your models (from Hugging Face). It will then be live on the web.

💻 How to Use the App
---------------------

1.  **Allow Permissions:** The app will ask for permission to use your webcam. Please allow it.
    
2.  **Input 1 (Face):** Click the **"Capture Face"** button to take a snapshot.
    
3.  **Input 2 (Voice):** Click **"Record/Upload Voice Clip"** to upload an .mp3 or .wav file of your voice.
    
4.  **Analyze:** Click the **"Analyze Environment"** button.
    
5.  **View Results:**
    
    *   **Cognitive Decision:** See the final fused emotion and the probability chart.
        
    *   **Virtual Home:** Watch the IoT dashboard update instantly based on your emotion.
