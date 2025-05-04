import os
import h5py
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuración ---
DATA_PATH = "BraTS2020/content/data/"
OUTPUT_DIR = "preprocessed_data"
IMG_SIZE = (128, 128)  # Tamaño de reescalado
BATCH_SIZE = 10        # Archivos por lote
SAMPLE_PLOTS = 3       # Muestras visuales a guardar

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

def load_h5_file(filepath):
    """Carga y preprocesa archivos .h5, convirtiendo máscaras multicanal a binarias"""
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Obtener modalidad FLAIR (o primera disponible)
            image = None
            for modality in ['flair', 't2', 't1ce', 't1']:
                if modality in f:
                    image = f[modality][()]
                    break
            if image is None and len(f.keys()) > 0:
                image = f[list(f.keys())[0]][()]  # Usar primera modalidad disponible

            # 2. Obtener máscara (priorizar 'seg' o 'mask')
            mask = None
            for mask_key in ['seg', 'mask']:
                if mask_key in f:
                    mask = f[mask_key][()]
                    break
            if mask is None and len(f.keys()) > 1:
                mask = f[list(f.keys())[1]][()]  # Usar segundo dataset si existe
            
            if image is None:
                raise ValueError("No se encontró dataset de imagen")

            # 3. Procesamiento de imagen
            image = resize(image, IMG_SIZE, order=1, anti_aliasing=True, preserve_range=True)
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
            
            # 4. Procesamiento de máscara (convertir multicanal a binario)
            if mask is not None:
                mask = resize(mask, IMG_SIZE, order=0, preserve_range=True)
                if mask.ndim == 3 and mask.shape[-1] > 1:  # Máscara multicanal
                    mask = (mask[..., 0] > 0.5).astype(np.float32)  # Usar solo primer canal o combinar canales
                else:
                    mask = (mask > 0.5).astype(np.float32)
                mask = mask[..., np.newaxis]  # Asegurar (H,W,1)
            else:
                mask = np.zeros_like(image[..., 0])[..., np.newaxis]  # Máscara vacía

            return image[..., np.newaxis], mask  # Imagen (H,W,1), Máscara (H,W,1)

    except Exception as e:
        print(f"⚠️ Error procesando {os.path.basename(filepath)}: {str(e)}")
        return None, None

def plot_sample(image, mask, save_path):
    """Visualización de muestra con superposición"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Imagen FLAIR
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title("MRI (FLAIR)")
        plt.axis('off')
        
        # Máscara superpuesta
        plt.subplot(1, 2, 2)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.imshow(mask.squeeze(), cmap='jet', alpha=0.5)
        plt.title("Segmentación")
        plt.axis('off')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        plt.close()
    except Exception as e:
        print(f"⚠️ Error al graficar: {str(e)}")

def main():
    print("⚡ Iniciando preprocesamiento (convertir multicanal a binario)...")
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.h5')]
    print(f"📂 Encontrados {len(files)} archivos .h5")

    for batch_idx in tqdm(range(0, len(files), BATCH_SIZE), desc="Procesando"):
        batch_files = files[batch_idx:batch_idx + BATCH_SIZE]
        X_batch, Y_batch = [], []
        
        for filename in batch_files:
            image, mask = load_h5_file(os.path.join(DATA_PATH, filename))
            
            if image is not None and mask is not None:
                X_batch.append(image)
                Y_batch.append(mask)
                
                # Guardar muestras del primer lote
                if batch_idx == 0 and len(X_batch) <= SAMPLE_PLOTS:
                    plot_sample(image, mask, f"{OUTPUT_DIR}/plots/sample_{len(X_batch)}.png")
        
        # Guardar lote procesado
        if X_batch:
            batch_id = batch_idx // BATCH_SIZE
            np.save(f"{OUTPUT_DIR}/X_batch_{batch_id}.npy", np.array(X_batch, dtype=np.float32))
            np.save(f"{OUTPUT_DIR}/Y_batch_{batch_id}.npy", np.array(Y_batch, dtype=np.float32))

    print(f"\n✅ Preprocesamiento completado. Datos guardados en '{OUTPUT_DIR}'")
    print(f"   - Imágenes: {X_batch[0].shape} (último lote)")
    print(f"   - Máscaras: {Y_batch[0].shape} (último lote)")

if __name__ == "__main__":
    main()