import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load color image
img = cv2.imread('iris.png', 1)

if img is None:
    print("Error: Image not found or unable to read.")
    exit()
else:
    print("Image loaded successfully!")
# 1. Padding
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )

    return padded_image
border_reflect = padding(img, 100)
cv2.imwrite('solutions/1. padding.png', border_reflect)
# 2. Cropping
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]

    return cropped_image
height, width = img.shape[:2]
cropped_image = crop(
    img,
    200,
    width - 130,
    200,
    height - 130
)
cv2.imwrite('solutions/2. Cropped Image.png', cropped_image)
# 3. Resize
def resize(image, width, height):
    resized_image = cv2.resize(image, (200, 200)                           )
    return resized_image
resized_image = resize(img, 200, 200)
cv2.imwrite('solutions/3. Resize.png', resized_image)
# 4. Manual copy
height, width, channels = img.shape
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
def copy(image, emptyPictureArray):
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]

    return emptyPictureArray
copied_image = copy(img, emptyPictureArray)
cv2.imwrite('solutions/4. Manual Copy.png', copied_image)
# 5. Grayscale
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
gray_image = grayscale(img)
cv2.imwrite('solutions/5. Grayscale.png', gray_image)

# 6. HSV
def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image
hsv_image = hsv(img)
cv2.imwrite('solutions/6. HSV.png', hsv_image)

# 7. Color shifting
def hue_shifted(image, emptyPictureArray, hue):
    emptyPictureArray[:] = image + hue
    return emptyPictureArray
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
hue_shifted_image = hue_shifted(img, emptyPictureArray, 50)
cv2.imwrite('solutions/7. Color Shifted.png', hue_shifted_image)

# 8. smoothing
def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15),0)
smoothed_image = smoothing(img)
cv2.imwrite('solutions/8. Smoothed.png', smoothed_image)
# 9. rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated_image = image
    return image

rotated_image = rotation(img, 180)
cv2.imwrite('solutions/9. Rotated.png', rotated_image)
