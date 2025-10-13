import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, util

def get_neighbors_indices(row, col, max_row, max_col):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < max_row and 0 <= nc < max_col:
                neighbors.append((nr, nc))
    return neighbors

def pca_noise_reduction(image, max_iterations=10, sigma=15):
    rows, cols = image.shape
    denoised_image = image.copy().astype(float)
    
    for iteration in range(max_iterations):
        new_image = denoised_image.copy()
        
        for i in range(rows):
            for j in range(cols):
                neighbors = get_neighbors_indices(i, j, rows, cols)
                
                weights = []
                intensities = []
                
                for nr, nc in neighbors:
                    diff = abs(denoised_image[nr, nc] - denoised_image[i, j])
                    weight = np.exp(-diff / sigma)
                    weights.append(weight)
                    intensities.append(denoised_image[nr, nc])
                
                weights = np.array(weights)
                intensities = np.array(intensities)
                
                if weights.sum() > 0:
                    new_value = np.sum(weights * intensities) / np.sum(weights)
                    new_image[i, j] = new_value
        
        denoised_image = new_image
    
    return denoised_image.astype(np.uint8)

# Load sample image and add noise
image = color.rgb2gray(data.astronaut())
image = (image * 255).astype(np.uint8)

noisy_image = util.random_noise(image, mode='s&p', amount=0.05)  # Fixed noise mode here
noisy_image = (noisy_image * 255).astype(np.uint8)

# Apply PCA-based noise reduction
denoised = pca_noise_reduction(noisy_image, max_iterations=10, sigma=20)

# Visualize results
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Denoised Image')
plt.imshow(denoised, cmap='gray')
plt.axis('off')

plt.show()
