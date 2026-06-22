# 🔬 SkinSense AI - Skin Disease Analyzer

**SkinSense AI** is a complete, AI-powered web application built using Python Flask. It allows users to upload a close-up image of a skin lesion, predicts the matching skin condition using an EfficientNetB0 Deep Learning model trained on the HAM10000 dataset, and suggests over-the-counter (OTC) medical products available in India along with precautions, doctor visitation recommendations, and a localized Tamil language summary using the **Groq API** with **LLaMA 3**.

---

## ⚠️ Important Safety & Training Scope Warnings
- **Dataset Scope Limit**: The deep learning model can only predict conditions that are explicitly present in the training dataset.
- **Acne Detection**: If the `Acne` class folder is not provided in the dataset during model training, the model cannot detect Acne. Uploading an unsupported condition like Acne will not result in a false alarm category; instead, the system will apply confidence safety filters.
- **Low Confidence Gating**:
  - **Confidence > 80%**: Confident prediction. Displays the category as a primary match.
  - **Confidence 60% - 80%**: Medium confidence. Handled as `Inconclusive / Medium Confidence` (scary names are hidden) and moves the matches to alternative badges.
  - **Confidence < 60%**: Low confidence. Handled as `Inconclusive / Low Confidence`, warning the user that the image likely belongs to an out-of-distribution condition (e.g. Acne or Eczema).

---

## 🌟 Key Features
- **AI-Powered Diagnostics**: Predicts 7 different types of skin conditions using transfer learning with an EfficientNetB0 base.
- **Dynamic File Uploader**: A modern dark-themed front end with drag-and-drop support, click-to-browse, and instant client-side image previewing.
- **Visual Confidence Indicators**: Results page featuring an animated progress bar that color-codes confidence levels (Green >80%, Amber 60-80%, Red <60%).
- **Llama 3 Clinical Context**: Calls Groq API (`llama3-8b-8192`) to fetch Descriptions, Causes, Precautions, and India-available OTC products.
- **Tamil Language Summaries**: Bridges local language barriers by including a summary in Tamil, rendered in a distinct card layout.
- **PDF Report Downloads**: Streamlined generation of clinical PDF reports containing prediction headers, patient images, confidences, and disclaimers using ReportLab.
- **Fail-Safe Offline Mode**: Full local simulation fallback database that runs seamlessly if the model hasn't been trained or if the Groq API key is missing.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python Flask (v3.0+) |
| **Deep Learning** | TensorFlow, Keras (EfficientNetB0) |
| **Image Preprocessing** | Pillow (PIL), NumPy |
| **LLM Inference** | Groq SDK (`llama3-8b-8192`) |
| **PDF Generation** | ReportLab |
| **Frontend UI** | HTML5, Bootstrap 5 CDN, FontAwesome, JavaScript (ES6) |
| **Styling & Theme** | Vanilla CSS (Teal `#1D9E75` dark theme, custom hover animations) |
| **Configuration** | Dotenv (`.env` file) |

---

## 📂 Project Structure

```text
SkinSense-AI/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── model/
│   └── train_model.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html
└── utils/
    ├── predict.py
    └── groq_helper.py
```

---

## 🚀 Setup & Installation

Follow these steps to run the application locally on Windows:

### 1. Clone or Copy Project Files
Place all files inside a single workspace folder, for example `SkinSense-AI/`.

### 2. Install Required Python Dependencies
Open PowerShell or Command Prompt inside the directory and install packages:
```bash
pip install -r requirements.txt
```

### 3. Setup Groq API Key
Create a `.env` file in the root folder (pre-created template is ready):
```env
GROQ_API_KEY=your_groq_api_key_here
```
> **Note**: You can obtain a free-tier API key by signing up at [Groq Console](https://console.groq.com/). If no key is set, the application automatically uses high-fidelity offline fallbacks.

### 4. How the Image Analysis Operates
Image analysis is performed in the cloud using **Groq's LLaMA 3.2 Multimodal Vision model (`llama-3.2-11b-vision-preview`)**.
* There is no local model file training required to run the primary web app.
* If no key is set or the API fails, the application automatically triggers a high-fidelity **Mock/Simulation mode** for testing layouts.
* The local deep learning training script (`model/train_model.py`) and evaluation script (`evaluate_model.py`) are preserved in the codebase as reference utilities but are not required for main app deployment.

### 5. Launch the Web Server
Start the Flask application:
```bash
python app.py
```
The app will run on:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📊 Neural Network Class Map
The model supports classification across 7 distinct skin lesions:
1. **Actinic Keratosis** (Precancerous solar spots)
2. **Basal Cell Carcinoma** (Slow-growing skin cancer)
3. **Benign Keratosis** (Seborrheic plaques / non-cancerous)
4. **Dermatofibroma** (Fibrous benign nodules)
5. **Melanocytic Nevi** (Common moles)
6. **Melanoma** (Highly malignant pigment cancer)
7. **Vascular Lesion** (Cherry angiomas / capillary spots)

---

## ⚠️ Disclaimer
SkinSense AI is developed solely for educational and training purposes. It is not designed to diagnose, treat, or prevent any skin disease. Always seek direct consultation from a medical professional or dermatologist for physical inspection.

---

## 👤 Author
* **Murali**
* AI & Data Science Student
* Paavai Engineering College, Tamil Nadu, India
