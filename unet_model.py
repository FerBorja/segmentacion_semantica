import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

def unet_model(input_size=(128, 128, 4), num_classes=1):
    """
    U-Net model for medical image segmentation
    
    Args:
        input_size: Tuple of (height, width, channels) for input images
        num_classes: Number of output classes (1 for binary, >1 for multi-class)
    """
    # --- Input Layer ---
    inputs = Input(input_size)
    
    # --- Encoder Path ---
    # Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # --- Bottleneck ---
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    
    # --- Decoder Path ---
    # Block 4
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    
    # Block 5
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    
    # --- Output Layer ---
    if num_classes == 1:
        # Binary segmentation
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
        loss = 'binary_crossentropy'
    else:
        # Multi-class segmentation
        outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c5)
        loss = 'sparse_categorical_crossentropy'
    
    # Build model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # Compile with appropriate settings
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss,
        metrics=['accuracy', MeanIoU(num_classes=num_classes)]
    )
    
    return model

if __name__ == "__main__":
    # Test the model
    print("Testing U-Net model creation...")
    
    # Binary segmentation model
    binary_model = unet_model(num_classes=1)
    binary_model.summary()
    print("\nBinary model output shape:", binary_model.output_shape)
    
    # Multi-class model (3 classes)
    multiclass_model = unet_model(num_classes=3)
    multiclass_model.summary()
    print("\nMulticlass model output shape:", multiclass_model.output_shape)
    
    print("\n✅ Model creation test completed successfully!")