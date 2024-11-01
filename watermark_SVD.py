import cv2 as cv
import numpy as np
from numpy.linalg import svd
from concurrent.futures import ThreadPoolExecutor
import numba
import argparse
from typing import Tuple
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Watermark Processing Tool')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--watermark', type=str, required=True,
                      help='Path to the watermark image')
    parser.add_argument('--alpha', type=float, default=0.0001,
                      help='Alpha value for watermark strength (default: 0.0001)')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for processing (default: 64)')
    parser.add_argument('--output', type=str, default='output.png',
                      help='Output path for watermarked image')
    parser.add_argument('--recovered-output', type=str, default='watermark_recovered.png',
                      help='Output path for recovered watermark')
    parser.add_argument('--noise-reduction', type=float, default=0.7,
                      help='Noise reduction threshold (0.0-1.0, default: 0.7)')
    return parser.parse_args()


@numba.jit(nopython=True)
def process_batch(image_batch: np.ndarray, watermark_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Procesa un lote de bloques de imagen de manera vectorizada
    """
    batch_size = image_batch.shape[0]
    results = np.zeros((batch_size, 4, 4), dtype=np.float64)
    
    for i in range(batch_size):
        U, S, V = svd(image_batch[i])
        S[3] = S[2]
        S[2] += alpha * watermark_values[i]
        results[i] = U @ np.diag(S) @ V
        
    return results

def prepare_batches(image: np.ndarray, watermark: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepara los lotes de datos para procesamiento vectorizado
    """
    wtm_x, wtm_y = watermark.shape
    indices = [(i, j) for i in range(0, wtm_x * 4, 4) 
              for j in range(0, wtm_y * 4, 4)]
    
    num_blocks = len(indices)
    num_batches = (num_blocks + batch_size - 1) // batch_size
    
    # Preparar arrays para almacenar los bloques
    image_blocks = np.zeros((num_blocks, 4, 4), dtype=np.float64)
    watermark_values = np.zeros(num_blocks, dtype=np.float64)
    
    # Llenar los arrays de manera vectorizada
    for idx, (i, j) in enumerate(indices):
        image_blocks[idx] = image[i:i + 4, j:j + 4]
        watermark_values[idx] = watermark[i//4][j//4]
    
    return image_blocks, watermark_values, indices

def make_watermark(image: np.ndarray, watermark: np.ndarray, alpha: float, batch_size: int) -> np.ndarray:
    """
    Aplica la marca de agua usando procesamiento por lotes y operaciones vectorizadas
    """
    img_x, img_y = image.shape
    wtm_x, wtm_y = watermark.shape

    if img_x < wtm_x * 4 or img_y < wtm_y * 4:
        raise ValueError("The input image must be 4x of the watermark")

    result = image.astype(np.float64)
    
    # Preparar datos en lotes
    image_blocks, watermark_values, indices = prepare_batches(image, watermark, batch_size)
    
    # Procesar por lotes
    num_blocks = len(indices)
    for batch_start in range(0, num_blocks, batch_size):
        batch_end = min(batch_start + batch_size, num_blocks)
        
        # Procesar lote actual
        batch_results = process_batch(
            image_blocks[batch_start:batch_end],
            watermark_values[batch_start:batch_end],
            alpha
        )
        
        # Actualizar resultado
        for idx, (i, j) in enumerate(indices[batch_start:batch_end]):
            result[i:i + 4, j:j + 4] = batch_results[idx]
    
    return result

@numba.jit(nopython=True)
def process_recovery_batch(image_batch: np.ndarray, alpha: float) -> np.ndarray:
    """
    Procesa un lote de bloques para recuperación de manera vectorizada
    """
    batch_size = image_batch.shape[0]
    results = np.zeros(batch_size, dtype=np.float64)
    
    for i in range(batch_size):
        _, S, _ = svd(image_batch[i])
        results[i] = 255.0 if S[2] - S[3] >= alpha else 0.0
    
    return results

def recover_watermark(image: np.ndarray, wtm_shape: tuple[int, int], alpha: float, batch_size: int) -> np.ndarray:
    """
    Recupera la marca de agua usando procesamiento por lotes
    """
    wtm_x, wtm_y = wtm_shape
    img_x, img_y = image.shape

    if img_x < wtm_x * 4 or img_y < wtm_y * 4:
        raise ValueError("The input image must be 4x of the watermark")
    
    result = np.zeros(wtm_shape, dtype=np.float64)
    
    # Preparar datos en lotes
    image_blocks = np.zeros(((wtm_x * wtm_y), 4, 4), dtype=np.float64)
    indices = [(i, j) for i in range(0, wtm_x * 4, 4) 
              for j in range(0, wtm_y * 4, 4)]
    
    # Llenar bloques de manera vectorizada
    for idx, (i, j) in enumerate(indices):
        image_blocks[idx] = image[i:i + 4, j:j + 4]
    
    # Procesar por lotes
    num_blocks = len(indices)
    for batch_start in range(0, num_blocks, batch_size):
        batch_end = min(batch_start + batch_size, num_blocks)
        
        # Procesar lote actual
        batch_results = process_recovery_batch(
            image_blocks[batch_start:batch_end],
            alpha
        )
        
        # Actualizar resultado
        for idx, (i, j) in enumerate(indices[batch_start:batch_end]):
            result[i//4][j//4] = batch_results[idx]
    
    return result

def enhance_recovered_watermark(recovered: np.ndarray, noise_threshold: float = 0.7) -> np.ndarray:
    """
    Mejora la calidad de la marca de agua recuperada reduciendo el ruido
    """
    # Normalizar la imagen
    recovered = cv.normalize(recovered, None, 0, 255, cv.NORM_MINMAX)
    
    # Aplicar desenfoque gaussiano suave para reducir ruido inicial
    blurred = cv.GaussianBlur(recovered, (3, 3), 0)
    
    # Aplicar umbral adaptativo con parámetros optimizados
    block_size = 11
    C = 2
    enhanced = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        block_size,
        C
    )
    
    # Reducción de ruido mediante operaciones morfológicas
    kernel = np.ones((2,2), np.uint8)
    enhanced = cv.morphologyEx(enhanced, cv.MORPH_OPEN, kernel)
    
    # Aplicar umbral basado en el parámetro de reducción de ruido
    threshold_value = int(255 * noise_threshold)
    _, enhanced = cv.threshold(
        enhanced,
        threshold_value,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    
    # Eliminar pequeños artefactos
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(enhanced, connectivity=8)
    min_size = 10  # Tamaño mínimo de componente a mantener
    
    # Crear máscara para componentes grandes
    mask = np.zeros_like(enhanced)
    for i in range(1, num_labels):  # Empezar desde 1 para ignorar el fondo
        if stats[i, cv.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 255
    
    # Aplicar la máscara
    enhanced = cv.bitwise_and(enhanced, mask)
    
    return enhanced

def process_and_save_output(image: np.ndarray, output_path: str):
    """
    Procesa y guarda una imagen, creando directorios si es necesario
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv.imwrite(output_path, image)

def main():
    args = parse_arguments()
    start_time = time.time()
    
    # Cargar imágenes
    wtm = cv.imread(args.watermark, cv.IMREAD_GRAYSCALE)
    _, wtm = cv.threshold(wtm, 10, 255, cv.THRESH_BINARY)
    img = cv.imread(args.image, cv.IMREAD_GRAYSCALE)
    
    print(f"Procesando imagen de {img.shape} con marca de agua de {wtm.shape}")
    
    # Aplicar marca de agua
    result = make_watermark(img, wtm, args.alpha, args.batch_size)
    show = result.astype(np.uint8)
    
    # Guardar imagen marcada
    process_and_save_output(show, args.output)
    print(f"Imagen marcada guardada en: {args.output}")
    
    # Recuperar y mejorar marca de agua
    result2 = recover_watermark(result, wtm.shape, args.alpha, args.batch_size).astype(np.uint8)
    
    # Aplicar mejoras adicionales antes del enhancement
    result2 = cv.normalize(result2, None, 0, 255, cv.NORM_MINMAX)
    
    # Mejorar la marca de agua recuperada
    enhanced_watermark = enhance_recovered_watermark(result2, args.noise_reduction)
    
    # Guardar marca de agua recuperada
    process_and_save_output(enhanced_watermark, args.recovered_output)
    print(f"Marca de agua recuperada guardada en: {args.recovered_output}")
    
    # Mostrar tiempo de ejecución
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
    
    # Mostrar resultados
    cv.imshow("Original con marca de agua", show)
    cv.waitKey(0)
    cv.imshow("Marca de agua recuperada", enhanced_watermark)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()