import cv2
import numpy as np

img = cv2.imread("input_image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Negative
negative = 255 - gray

# Log Transformation
c = 255 / np.log(1 + 255)
log_img = c * np.log(1 + gray.astype(np.float32))
log_img = np.uint8(log_img)

# Gamma Transformation
gamma = 0.5
gamma_img = 255 * ((gray / 255) ** gamma)
gamma_img = np.uint8(gamma_img)

cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Negative", negative)
cv2.imshow("Log", log_img)
cv2.imshow("Gamma", gamma_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
