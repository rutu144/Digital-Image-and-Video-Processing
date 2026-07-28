import cv2
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("input_image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Select row number
row = 100

# Extract row profile
row_profile = image[row, :]

# Display image
plt.imshow(image, cmap="gray")
plt.title("Input Image")
plt.axis("off")
plt.show()

# Plot row profile
plt.figure(figsize=(10,4))
plt.plot(row_profile)
plt.title("Row Profile")
plt.xlabel("Column Number")
plt.ylabel("Pixel Intensity")
plt.grid(True)
plt.show()
