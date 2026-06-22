import os
import json
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
CUSTOM_DATASET_DIR = './data/skin_dataset'
HAM10000_DIR = './data/HAM10000'
MODEL_SAVE_PATH = 'model/skinsense_model.h5'
LABELS_SAVE_PATH = 'model/class_labels.json'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
STAGE_1_EPOCHS = 5
STAGE_2_EPOCHS = 15

def get_generators():
    """
    Sets up the training and validation generators.
    Prioritizes custom dataset directory format (data/skin_dataset/train and data/skin_dataset/val).
    Falls back to HAM10000 (data/HAM10000/) with 20% validation split.
    """
    # 1. Check for custom dataset format
    custom_train_path = os.path.join(CUSTOM_DATASET_DIR, 'train')
    custom_val_path = os.path.join(CUSTOM_DATASET_DIR, 'val')
    
    if os.path.exists(custom_train_path) and os.path.exists(custom_val_path):
        print(f"[SkinSense AI] Using custom dataset directory: {CUSTOM_DATASET_DIR}")
        
        # Generator settings - using preprocessing_function instead of rescale 1/255
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            preprocessing_function=preprocess_input
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        train_generator = train_datagen.flow_from_directory(
            custom_train_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            custom_val_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # Crucial for classification report & confusion matrix evaluation
        )
        
        return train_generator, validation_generator
        
    # 2. Check for HAM10000 dataset format
    elif os.path.exists(HAM10000_DIR):
        print(f"[SkinSense AI] Using HAM10000 dataset directory: {HAM10000_DIR}")
        
        # Generator settings - using validation split
        datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            HAM10000_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            HAM10000_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False  # Crucial for classification report & confusion matrix evaluation
        )
        
        return train_generator, validation_generator
        
    else:
        print("[SkinSense AI] Error: No valid dataset found.")
        print(f"Please place your dataset in either '{CUSTOM_DATASET_DIR}/' (with train/ and val/ subdirectories)")
        print(f"or in '{HAM10000_DIR}/' (with class directories).")
        return None, None

def train():
    print("TensorFlow Version:", tf.__version__)
    
    # Set up generators
    train_gen, val_gen = get_generators()
    if train_gen is None or val_gen is None:
        return
        
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    
    # Save class indices / labels map to JSON
    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    os.makedirs(os.path.dirname(LABELS_SAVE_PATH), exist_ok=True)
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_labels, f, indent=4)
    print(f"[SkinSense AI] Class labels saved to {LABELS_SAVE_PATH}: {class_labels}")
    
    # Calculate Class Weights to handle class imbalance
    print("Calculating class weights...")
    class_counts = Counter(train_gen.classes)
    total_samples = len(train_gen.classes)
    class_weights = {}
    for class_id, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_id] = weight
        print(f" - Class '{class_labels[class_id]}': Count={count}, Weight={weight:.4f}")
        
    # Build Model (EfficientNetB0 Transfer Learning)
    print("[SkinSense AI] Initializing EfficientNetB0 pre-trained base...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Create top classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # --- STAGE 1: Freeze base and train classifier head ---
    print("\n[SkinSense AI] STAGE 1: Training top classifier head (base model frozen)...")
    base_model.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Standard Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
    
    history_stage1 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=STAGE_1_EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    
    # --- STAGE 2: Unfreeze last 30 layers and fine-tune ---
    print("\n[SkinSense AI] STAGE 2: Fine-tuning last 30 layers of the base model...")
    base_model.trainable = True
    
    # Freeze all layers except the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Re-train with lower learning rate
    history_stage2 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=STAGE_2_EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    
    print("\n[SkinSense AI] Training completed! Evaluating best model...")
    
    # Load the best saved model for final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        best_model = load_model(MODEL_SAVE_PATH)
    else:
        best_model = model
        
    # Print training metrics
    # Get final epoch stats
    s2_epochs_run = len(history_stage2.history['accuracy'])
    final_train_acc = history_stage2.history['accuracy'][-1]
    final_val_acc = history_stage2.history['val_accuracy'][-1]
    print(f"Final Epoch Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Epoch Validation Accuracy: {final_val_acc:.4f}")
    
    # Make predictions for evaluation reports
    print("[SkinSense AI] Generating predictions on validation set...")
    val_gen.reset()
    y_pred_probs = best_model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"\nModel and labels successfully saved to {MODEL_SAVE_PATH} and {LABELS_SAVE_PATH}.")

if __name__ == '__main__':
    train()
