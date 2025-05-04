import matplotlib.pyplot as plt

# Simulación de historial de entrenamiento (reemplaza con tu historial real)
epochs = list(range(1, 51))
train_iou = [0.5 + 0.005 * i for i in epochs]  # Mejora progresiva
val_iou = [0.48 + 0.0048 * i for i in epochs]
train_acc = [0.7 + 0.006 * i for i in epochs]
val_acc = [0.68 + 0.0055 * i for i in epochs]

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_iou, label='Train IoU', color='green', linestyle='-')
plt.plot(epochs, val_iou, label='Val IoU', color='green', linestyle='--')
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', linestyle='-')
plt.plot(epochs, val_acc, label='Val Accuracy', color='blue', linestyle='--')

plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.ylim(0.4, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar imagen
plt.savefig('assets/images/training_progress.png')
plt.show()
