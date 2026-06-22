import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Try to import matplotlib and handle cases where it might fail in CLI environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Set non-interactive backend for headless CLI servers
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
CUSTOM_DATASET_DIR = './data/skin_dataset'
HAM10000_DIR = './data/HAM10000'
MODEL_PATH = 'model/skinsense_model.h5'
LABELS_PATH = 'model/class_labels.json'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EVAL_PLOT_PATH = 'static/evaluation/confusion_matrix.png'

def get_validation_generator():
    """
    Checks for the dataset path and loads the validation data generator
    exactly matching the training preprocess settings.
    """
    custom_val_path = os.path.join(CUSTOM_DATASET_DIR, 'val')
    
    if os.path.exists(custom_val_path):
        print(f"[SkinSense AI] Loading validation set from custom path: {custom_val_path}")
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        return datagen.flow_from_directory(
            custom_val_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    elif os.path.exists(HAM10000_DIR):
        print(f"[SkinSense AI] Loading validation set from HAM10000: {HAM10000_DIR}")
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )
        return datagen.flow_from_directory(
            HAM10000_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    else:
        print("[SkinSense AI] Error: No valid dataset found to evaluate.")
        print(f"Please put validation data in '{custom_val_path}' or HAM10000 in '{HAM10000_DIR}'.")
        return None

def evaluate():
    # 1. Check for model and labels
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(f"[SkinSense AI] Error: Model '{MODEL_PATH}' or labels mapping '{LABELS_PATH}' does not exist.")
        print("Please run training first (python model/train_model.py) before evaluating.")
        return
        
    print("[SkinSense AI] Loading class labels mapping...")
    with open(LABELS_PATH, 'r') as f:
        data = json.load(f)
        class_labels = {int(k): v for k, v in data.items()}
        class_names = [class_labels[i] for i in sorted(class_labels.keys())]
        
    print(f"[SkinSense AI] Loading trained model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    # 2. Get generator
    val_generator = get_validation_generator()
    if val_generator is None:
        return
        
    # Verify mapping aligns with names
    num_classes = len(class_names)
    generator_classes = list(val_generator.class_indices.keys())
    if num_classes != len(generator_classes):
        print("[SkinSense AI] WARNING: Saved class labels count does not match the loaded validation folders.")
        class_names = generator_classes
        
    # 3. Perform prediction
    print("[SkinSense AI] Generating predictions on validation dataset...")
    val_generator.reset()
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes
    
    # 4. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted Precision:  {precision:.4f}")
    print(f"Weighted Recall:     {recall:.4f}")
    print(f"Weighted F1-Score:   {f1:.4f}")
    print("====================================================\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("Confusion Matrix:")
    print(cm)
    
    # 5. Plot and save confusion matrix
    if HAS_MATPLOTLIB:
        print(f"[SkinSense AI] Generating confusion matrix plot...")
        os.makedirs(os.path.dirname(EVAL_PLOT_PATH), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        # Plot with blue color mapping
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())
        plt.title('SkinSense AI - Confusion Matrix')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(EVAL_PLOT_PATH, dpi=150)
        plt.close()
        print(f"[SkinSense AI] Confusion matrix plot saved to: {EVAL_PLOT_PATH}")
    else:
        print("[SkinSense AI] Matplotlib not available, skipping confusion matrix plot export.")

if __name__ == '__main__':
    evaluate()
