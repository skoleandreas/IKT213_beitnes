import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load images
img = cv2.imread('lambo.png', 1)
img1 = cv2.imread('shapes.png', 0)
template = cv2.imread('shapes_template.jpg', 0)

if img1 is None:
    print("Error: Image not found or unable to read.")
    exit()
else:
    print("Image loaded successfully!")

if (template is None):
    print("Error: Image not found or unable to read.")
    exit()
else:
    print("Image loaded successfully!")
if img is None:
    print("Error: Image not found or unable to read.")
    exit()
else:
    print("Image loaded successfully!")

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(src=blur, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobel_y = cv2.Sobel(src=blur, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel)
sobel_image = sobel_edge_detection(img)
cv2.imwrite("solutions/1. sobel_lambo.png", sobel_image)

def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, threshold_1, threshold_2)
    return cv2.convertScaleAbs(canny)
canny_image = canny_edge_detection(img)
cv2.imwrite("solutions/2. canny_lambo.png", canny_image)


def template_match(image, template):
    match_result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    matched_locations = np.where(match_result >= threshold)
    template_height, template_width = template.shape
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for location in zip(*matched_locations[::-1]):
        top_left = location
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        cv2.rectangle(result_image, top_left, bottom_right,(0, 0, 255), 2)
    return result_image

template_result = template_match(img1, template)
cv2.imwrite("solutions/3. template_match.png", template_result)

def resize(image, scale_factor: int, up_or_down: str):
    rows, cols = image.shape[:2]
    if up_or_down == "up":
        resized_image = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
    elif up_or_down == "down":
        resized_image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")
    return resized_image
resized_image1 = resize(img, scale_factor=2, up_or_down="up")
resized_image2 = resize(img, scale_factor=2, up_or_down="down")
cv2.imwrite("solutions/4. Big_resized_lambo.png", resized_image1)
cv2.imwrite("solutions/4. Small_resized_lambo.png", resized_image2)