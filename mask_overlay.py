import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen y máscara desde las rutas correctas
image = cv2.imread('assets/images/image.png')
mask = cv2.imread('assets/images/mask.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error al cargar la imagen. Verifica la ruta del archivo.")
if mask is None:
    print("Error al cargar la máscara. Verifica la ruta del archivo.")

# Asegurarse de que la imagen y la máscara tienen el mismo tamaño
if image.shape != mask.shape:
    print(f"Tamaño de la imagen: {image.shape}, tamaño de la máscara: {mask.shape}")
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Ajustar la máscara

# Crear una máscara de color para superponerla
colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

# Superponer la máscara sobre la imagen original
overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

# Guardar la imagen resultante
cv2.imwrite('assets/images/mask_overlay_example.png', overlay)

# Mostrar la imagen superpuesta
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')  # No mostrar los ejes
plt.show()
