import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import os
import datetime
from unet_model import unet_model

# --- Configuration ---
BATCH_SIZE = 8
EPOCHS = 50
INPUT_SHAPE = (128, 128, 4)  # 4 channels for BraTS modalities
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# --- Callbacks ---
def get_callbacks():
    return [
        EarlyStopping(patience=10, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

# --- Data Loading ---
def load_data():
    """Load and preprocess data with automatic channel detection"""
    try:
        X_train = np.load("preprocessed_data/X_batch_0.npy")
        Y_train = np.load("preprocessed_data/Y_batch_0.npy")
        
        # Remove singleton dimensions if they exist
        X_train = np.squeeze(X_train)
        Y_train = np.squeeze(Y_train)
        
        # Ensure proper shapes
        if X_train.ndim == 3:
            X_train = X_train[..., np.newaxis]
        if Y_train.ndim == 3:
            Y_train = Y_train[..., np.newaxis]
            
        # Determine number of classes from mask shape
        num_classes = Y_train.shape[-1] if Y_train.shape[-1] > 1 else 1
        
        print(f"📊 Data shapes - X: {X_train.shape}, Y: {Y_train.shape}")
        print(f"🔍 Detected number of classes: {num_classes}")
        
        # Train/validation split
        split_idx = int(0.8 * len(X_train))
        return (
            X_train[:split_idx], 
            Y_train[:split_idx],
            X_train[split_idx:], 
            Y_train[split_idx:],
            num_classes
        )
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# --- Training ---
def train():
    print("⚡ Starting training...")
    
    # Load data and detect number of classes
    X_train, Y_train, X_val, Y_val, num_classes = load_data()
    
    # Create model
    model = unet_model(input_size=INPUT_SHAPE, num_classes=num_classes)
    
    print("\n🧠 Model Summary:")
    model.summary()
    
    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    print("🚀 BraTS Segmentation Training Script")
    print("------------------------------------")
    
    try:
        history = train()
        print("\n✅ Training completed successfully!")
        print(f"TensorBoard logs: {LOG_DIR}")
        print(f"Best model saved to: best_model.h5")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")