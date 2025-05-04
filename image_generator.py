import numpy as np
import matplotlib.pyplot as plt
import cv2

# Crear una imagen de ejemplo (una imagen de MRI simple en escala de grises)
image = np.random.rand(256, 256)  # Imagen 256x256 con valores aleatorios
image = (image * 255).astype(np.uint8)  # Convertir a valores enteros de 0-255

# Crear una máscara simulada (simula la segmentación de una tumoración)
mask = np.zeros_like(image)
mask[80:150, 80:150] = 255  # Colocar un "tumor" en una parte de la imagen (cuadro blanco)

# Guardar imagen y máscara
cv2.imwrite('assets/images/image.png', image)
cv2.imwrite('assets/images/mask.png', mask)

# Visualizar y guardar las imágenes para verificar
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.savefig('assets/images/image_example.png')

plt.imshow(mask, cmap='gray')
plt.title('Máscara de Segmentación')
plt.axis('off')
plt.savefig('assets/images/mask_example.png')

# Mostrar las imágenes generadas
plt.show()
