import os
import cv2
import numpy as np
import shutil
from PIL import Image
from pathlib import Path

def resize_to_vizwiz_quality(image_np, max_dim=640):
    """Reduce las dimensiones para igualar la calidad típica de las cámaras de VizWiz."""
    h, w = image_np.shape[:2]
    if max(h, w) <= max_dim:
        return image_np
    
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_motion_blur(image_np, kernel_size=15, angle=0):
    """Aplica desenfoque de movimiento direccional."""
    kernel = np.zeros((kernel_size, kernel_size))
    center = int((kernel_size - 1) / 2)
    kernel[center, :] = np.ones(kernel_size)
    
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    
    return cv2.filter2D(image_np, -1, kernel)

def save_with_jpeg_compression(image_np, output_path, quality=30):
    """Guarda la imagen con compresión JPEG para generar artefactos."""
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    pil_img.save(str(output_path), "JPEG", quality=quality)
    
def apply_overexposure(image_np, factor=1.8, offset=50):
    """Simula sobreexposición quemando los píxeles hacia el blanco."""
    bright_img = image_np.astype(np.float32) * factor + offset
    return np.clip(bright_img, 0, 255).astype(np.uint8)

def main():
    img_dir = Path("./final_images")
    out_dir = Path("./")
    
    # Buscar una imagen válida en la carpeta
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions]
    
    if not image_paths:
        print("[!] No se han encontrado imágenes en ./final_images")
        return
    
    sorted_images = sorted(image_paths, key=lambda p: p.name)
        
    sample_img_path = sorted_images[8]
    print(f"Generando muestras visuales para: {sample_img_path.name}...")
    
    img = cv2.imread(str(sample_img_path))
    if img is None:
        print(f"[!] Error al leer la imagen {sample_img_path}")
        return
        
    rng = np.random.default_rng(42) # Semilla fija para reproducibilidad
    

    standard_img = resize_to_vizwiz_quality(img, max_dim=500)
    
    # Instead of shutil.copy2, we save the resized numpy array
    cv2.imwrite(str(out_dir / "sample_0_original.jpg"), standard_img)

    # Applying transforms to the downscaled image
    mod_img = apply_motion_blur(standard_img, kernel_size=12, angle=np.random.randint(0, 180))
    mod_img = apply_overexposure(mod_img, factor=(0.2*rng.random()+1.0), offset=20*rng.random())
    save_with_jpeg_compression(mod_img, out_dir / "sample_1_moderate.jpg", quality=60)

    if np.random.rand() > 0.5:
        sev_img = apply_overexposure(standard_img, factor=2.75, offset=100)
    else:
        sev_img = apply_motion_blur(standard_img, kernel_size=75, angle=np.random.randint(0, 180))
        
    save_with_jpeg_compression(sev_img, out_dir / "sample_2_severe.jpg", quality=40)
    
    print("\n¡Muestras generadas con éxito en el directorio actual (./)!")
    print("Descarga estos 3 archivos para comprobar las transformaciones:")
    print(" -> sample_0_original.jpg")
    print(" -> sample_1_moderate.jpg")
    print(" -> sample_2_severe.jpg")

if __name__ == "__main__":
    main()