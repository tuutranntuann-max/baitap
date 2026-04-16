import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def augment_image(image, task_type='bai1'):
    # 1. Resize (Áp dụng cho tất cả bài) [cite: 6, 13, 19, 24]
    img = cv2.resize(image, (224, 224))

    # 2. Thực hiện Augmentation
    if task_type == 'bai1' or task_type == 'bai4':
        # Lật ngang [cite: 7, 25]
        img = cv2.flip(img, 1)
        # Chỉnh độ sáng +/- 20% [cite: 8, 25]
        value = random.uniform(0.8, 1.2)
        img = np.clip(img * value, 0, 255).astype(np.uint8)
        # Chuyển Grayscale [cite: 9, 27]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif task_type == 'bai2':
        # Thêm Gaussian Noise
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    # 3. Chuẩn hóa về [0, 1] [cite: 10, 16, 21, 29]
    img_normalized = img.astype('float32') / 255.0
    return img_normalized

# Ví dụ hiển thị Bài 4 (1 gốc, 3 augmented)
original = cv2.imread('noithat.jpg')

# Check if the image was loaded successfully
if original is None:
    print("Error: Could not load image 'noithat.jpg'. Please ensure the file exists and the path is correct.")
else:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title("Gốc")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    for i in range(3):
        aug = augment_image(original, task_type='bai4')
        plt.subplot(1, 4, i+2)
        plt.title(f"Aug {i+1}")
        plt.imshow(aug, cmap='gray')
    plt.show()
    
