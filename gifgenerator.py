import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import imageio
import glob
import os

# Configuración
MODEL_PATH = "best_model.h5"
TEST_DATA_PATH = "preprocessed_data/X_batch_0.npy"
TEST_MASKS_PATH = "preprocessed_data/Y_batch_0.npy"
GIF_PATH = "tumor_segmentation.gif"
SAMPLE_SLICES = 10  # Número de cortes a incluir
FPS = 1  # Cuadros por segundo

def load_and_prepare_data():
    """Carga y prepara los datos con la forma correcta"""
    X_test = np.load(TEST_DATA_PATH)
    Y_test = np.load(TEST_MASKS_PATH)
    
    print(f"Forma original de X_test: {X_test.shape}")
    print(f"Forma original de Y_test: {Y_test.shape}")

    # Eliminar dimensiones unitarias si existen
    X_test = np.squeeze(X_test)
    Y_test = np.squeeze(Y_test)

    # Asegurar que los datos tengan la forma (slices, height, width, channels)
    if X_test.ndim == 5:  # Volúmenes completos
        X_test = X_test[0, :SAMPLE_SLICES]
        Y_test = Y_test[0, :SAMPLE_SLICES]
    elif X_test.ndim == 4:
        X_test = X_test[:SAMPLE_SLICES]
        Y_test = Y_test[:SAMPLE_SLICES]
    else:
        raise ValueError(f"Forma de datos no soportada: {X_test.shape}")

    print(f"Forma corregida de X_test: {X_test.shape}")
    print(f"Forma corregida de Y_test: {Y_test.shape}")
    
    return X_test, Y_test

def generate_gif(model, X_test, Y_test):
    """Genera el GIF comparativo"""
    Y_pred = model.predict(X_test, batch_size=2)
    print(f"Forma de predicciones: {Y_pred.shape}")

    images_for_gif = []
    for i in range(len(X_test)):
        fig = plt.figure(figsize=(15, 5))

        # Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i, ..., 0], cmap='gray')
        plt.title(f"Slice {i+1}")
        plt.axis('off')

        # Máscara real
        plt.subplot(1, 3, 2)
        plt.imshow(X_test[i, ..., 0], cmap='gray')
        plt.imshow(np.squeeze(Y_test[i]), alpha=0.4, cmap='Reds')
        plt.title("Ground Truth")
        plt.axis('off')

        # Predicción
        plt.subplot(1, 3, 3)
        plt.imshow(X_test[i, ..., 0], cmap='gray')
        if Y_pred.shape[-1] == 1:  # Binario
            pred = (Y_pred[i, ..., 0] > 0.5).astype(float)
        else:  # Multiclase
            pred = np.argmax(Y_pred[i], axis=-1)
        plt.imshow(pred, alpha=0.4, cmap='Greens')
        plt.title("Prediction")
        plt.axis('off')

        # Guardar y agregar al GIF
        temp_path = f"temp_slice_{i}.png"
        plt.tight_layout()
        plt.savefig(temp_path, bbox_inches='tight', dpi=100)
        plt.close()
        images_for_gif.append(imageio.imread(temp_path))

    imageio.mimsave(GIF_PATH, images_for_gif, fps=FPS)

    # Limpiar archivos temporales
    for file in glob.glob("temp_slice_*.png"):
        os.remove(file)

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    print(f"\nModelo cargado. Forma de entrada esperada: {model.input_shape}")

    X_test, Y_test = load_and_prepare_data()

    generate_gif(model, X_test, Y_test)
    print(f"\n✅ GIF generado exitosamente en: {os.path.abspath(GIF_PATH)}")
