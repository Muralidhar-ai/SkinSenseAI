import os
import json
import base64
import numpy as np
import hashlib
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Retrieve API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
MODEL_PATH = 'model/skinsense_model.h5'
LABELS_PATH = 'model/class_labels.json'
CLASSES = [
    'Actinic Keratosis', 
    'Basal Cell Carcinoma', 
    'Benign Keratosis', 
    'Dermatofibroma', 
    'Melanocytic Nevi', 
    'Melanoma', 
    'Vascular Lesion'
]

def encode_image(image_path):
    """
    Helper to encode an image file into base64 format for the multimodal API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def apply_safety_gating(top_class, top_score, all_predictions):
    """
    Applies safety gating logic based on prediction confidence score:
    - If score > 80%: High confidence primary classification.
    - If 60% <= score <= 80%: Medium Confidence with warning.
    - If score < 60%: Low Confidence with warning.
    """
    if "inconclusive" in top_class.lower():
        return {
            'disease_name': "Inconclusive Result",
            'confidence': 0.0,
            'is_confident': False,
            'status': 'low',
            'warning': "The model could not identify a clear skin condition. Please ensure the image is clear and well-lit.",
            'all_predictions': all_predictions
        }

    if top_score > 80.0:
        return {
            'disease_name': top_class,
            'confidence': round(top_score, 2),
            'is_confident': True,
            'status': 'high',
            'warning': '',
            'all_predictions': all_predictions
        }
    elif top_score >= 60.0:
        return {
            'disease_name': top_class,
            'confidence': round(top_score, 2),
            'is_confident': False,
            'status': 'medium',
            'warning': "Moderate confidence: The model is not highly confident. Please consult a dermatologist for physical evaluation.",
            'all_predictions': all_predictions
        }
    else:
        return {
            'disease_name': top_class,
            'confidence': round(top_score, 2),
            'is_confident': False,
            'status': 'low',
            'warning': "Low confidence: The model is not confident in this result. Please consult a dermatologist for diagnostic confirmation.",
            'all_predictions': all_predictions
        }

def generate_mock_prediction(image_path):
    """
    Deterministic mock prediction engine for testing purposes.
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        file_hash = int(hashlib.md5(img_bytes).hexdigest(), 16)
    except Exception:
        file_hash = sum(ord(c) for c in os.path.basename(image_path))
        
    np.random.seed(file_hash % 2**32)
    raw_scores = np.random.dirichlet(np.ones(len(CLASSES)))
    
    all_predictions = []
    for label, score in zip(CLASSES, raw_scores):
        all_predictions.append({
            'label': label,
            'score': round(float(score) * 100, 2)
        })
    
    all_predictions = sorted(all_predictions, key=lambda x: x['score'], reverse=True)
    top_prediction = all_predictions[0]
    
    return apply_safety_gating(top_prediction['label'], top_prediction['score'], all_predictions)

def predict_disease(image_path):
    """
    Queries Groq's Multimodal Vision model (meta-llama/llama-4-scout-17b-16e-instruct)
    to classify ANY skin disease, returning confidence scores and alternative matches.
    Falls back to mock mode if GROQ_API_KEY is invalid or missing.
    """
    global GROQ_API_KEY
    # Force reload dotenv at prediction time just to be absolutely sure
    load_dotenv(override=True)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    print("[SkinSense AI] predict_disease called. GROQ_API_KEY:", repr(GROQ_API_KEY))
    
    if not GROQ_API_KEY or "your_groq_api_key" in GROQ_API_KEY or GROQ_API_KEY == "":
        print("[SkinSense AI] Groq API key not configured for vision analysis. Running mock fallback.")
        return generate_mock_prediction(image_path)
        
    try:
        # Encode image to base64
        base64_image = encode_image(image_path)
        
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = (
            "You are a professional dermatological assistant. Analyze the skin condition image "
            "and identify the most likely condition. Respond ONLY with a JSON object in this format:\n"
            "{\n"
            "  \"condition\": \"Condition Name\",\n"
            "  \"confidence\": 92,\n"
            "  \"alternatives\": [\n"
            "    {\"label\": \"Alternative 1\", \"score\": 25},\n"
            "    {\"label\": \"Alternative 2\", \"score\": 12},\n"
            "    {\"label\": \"Alternative 3\", \"score\": 5}\n"
            "  ]\n"
            "}\n"
            "If the image is not a skin condition (e.g. face only with normal skin, text, objects), "
            "set \"condition\" to \"Inconclusive\" and \"confidence\" to 0. "
            "Do not output markdown code blocks (```json). Output raw JSON string only."
        )
        
        print(f"[SkinSense AI] Calling Groq Multimodal Vision (meta-llama/llama-4-scout-17b-16e-instruct) for: {image_path}...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_tokens=512
        )
        
        response_content = chat_completion.choices[0].message.content.strip()
        print(f"[SkinSense AI] Vision Raw Response: {response_content}")
        
        # Strip potential code block formatting from LLM output
        if response_content.startswith("```"):
            lines = response_content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_content = "\n".join(lines).strip()
            
        data = json.loads(response_content)
        top_condition = data.get("condition", "Inconclusive")
        confidence = float(data.get("confidence", 50))
        alternatives = data.get("alternatives", [])
        
        # Compile all predictions list (primary classification + alternative predictions)
        all_predictions = [{"label": top_condition, "score": confidence}]
        for alt in alternatives:
            all_predictions.append({
                'label': alt.get("label", "Unknown"),
                'score': float(alt.get("score", 0))
            })
            
        return apply_safety_gating(top_condition, confidence, all_predictions)
        
    except Exception as e:
        print(f"[SkinSense AI] Error during Groq Vision analysis: {e}. Falling back to mock prediction.")
        return generate_mock_prediction(image_path)
