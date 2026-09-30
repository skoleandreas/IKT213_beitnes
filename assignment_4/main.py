print("I used ORB for feature matching and RANSAC homography for image alignment.")
import cv2
import numpy as np
from PIL import Image

def harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    harris = reference_image.copy()
    harris[corners > 0.01 * corners.max()] = [0, 0, 255]
    cv2.imwrite("solutions/harris.png", harris)

def align_images(image_to_align, reference_image, max_features, good_match_precent):
    gray1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_precent)
    matches = matches[:num_good_matches]

    match_image = cv2.drawMatches(
        image_to_align, keypoints1,
        reference_image, keypoints2,
        matches, None
    )
    cv2.imwrite("solutions/matches.png", match_image)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt

    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = reference_image.shape[:2]
    aligned = cv2.warpPerspective(
        image_to_align, homography, (width, height)
    )
    cv2.imwrite("solutions/aligned.png", aligned)
    return aligned, homography

if __name__ == "__main__":
    reference_image = cv2.imread("reference_img.png")
    image_to_align = cv2.imread("align_this.jpg")
    harris_corner_detection(reference_image)
    aligned, homography = align_images(
        image_to_align,
        reference_image,
        max_features=1500,
        good_match_precent=0.15
    )
    pages = [
        Image.open(filename).convert("RGB")
        for filename in ["solutions/harris.png", "solutions/aligned.png", "solutions/matches.png"]
    ]
    pages[0].save(
        "solutions/assignment_4.pdf",
        save_all=True,
        append_images=pages[1:]
    )