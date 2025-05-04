import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# --- Configuración ---
MODEL_PATH = "best_model.h5"
SAMPLE_INDEX = 0  # Índice de la muestra a visualizar

# --- Funciones de evaluación ---
def plot_sample_prediction(model, X, y_true, index=0):
    """Visualiza una predicción junto con la verdad del terreno"""
    y_pred = model.predict(X[np.newaxis, index])
    y_pred_thresh = (y_pred > 0.5).astype(np.float32)  # Umbralización
    
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(X[index].squeeze(), cmap='gray')
    plt.title("Input MRI")
    plt.axis('off')
    
    # Máscara verdadera
    plt.subplot(1, 3, 2)
    plt.imshow(y_true[index].squeeze(), cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Predicción
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_thresh.squeeze(), cmap='jet')
    plt.title("Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("prediction_sample.png")
    plt.show()

def plot_training_history(history):
    """Grafica la evolución del entrenamiento"""
    plt.figure(figsize=(12, 4))
    
    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Exactitud
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

# --- Evaluación principal ---
def evaluate():
    # Cargar modelo y datos
    model = load_model(MODEL_PATH)
    X_val = np.load("preprocessed_data/X_batch_0.npy")[-10:]  # Usar últimas 10 muestras para evaluación
    Y_val = np.load("preprocessed_data/Y_batch_0.npy")[-10:]
    
    if X_val.ndim == 3:
        X_val = np.expand_dims(X_val, axis=-1)
        Y_val = np.expand_dims(Y_val, axis=-1)
    
    # Evaluación cuantitativa
    results = model.evaluate(X_val, Y_val, verbose=0)
    print("\n📝 Métricas de evaluación:")
    print(f"Pérdida: {results[0]:.4f}")
    print(f"Exactitud: {results[1]:.4f}")
    print(f"IoU: {results[2]:.4f}")
    
    # Visualización
    plot_sample_prediction(model, X_val, Y_val, SAMPLE_INDEX)
    
    # Si hay historial de entrenamiento
    try:
        history = np.load("training_history.npy", allow_pickle=True).item()
        plot_training_history(history)
    except:
        print("⚠️ No se encontró historial de entrenamiento")

if __name__ == "__main__":
    print("🔍 Evaluando modelo...")
    evaluate()