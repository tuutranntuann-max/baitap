original = cv2.imread(image_filename)

# Check if the image was loaded successfully
if original is None:
    print(f"Error: Could not load image '{image_filename}'. Please ensure the file exists and the path is correct.")
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
