import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

def rotate_to_horizontal(image):
    # Rotate the image to horizontal (90 degrees counterclockwise)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated_image

def rotate_to_vertical(image):
    # Rotate the image back to vertical (90 degrees clockwise)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def plot_images(images):
    num_images = len(images)
    
    for i, image in enumerate(images):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Image {i+1}/{num_images}")
        plt.axis('off')
        plt.show()

def image_stitching(images):
    
    images = [image.astype(np.uint8) for image in images]
    images = [rotate_to_horizontal(image) for image in images]
    images = list(reversed(images))
    
    if len(images) == 15:
        images = images[2:12]
    
    stitched_images = []

    # Accumulated homography matrix for stitching
    accumulated_homography = np.eye(3)

    # Iterate through pairs of adjacent images and stitch them together
    for i in range(len(images) - 1):
        # Perform keypoint and feature descriptor extraction
        orb = cv2.ORB_create()
        keypoints_and_descriptors = [orb.detectAndCompute(image, None) for image in [images[i], images[i + 1]]]

        # Match the keypoints using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(keypoints_and_descriptors[0][1], keypoints_and_descriptors[1][1])

        # Filter the matches to remove outliers using RANSAC
        src_pts = np.float32([keypoints_and_descriptors[0][0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_and_descriptors[1][0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Accumulate the homography matrices
        accumulated_homography = np.dot(M, accumulated_homography)

        # Warp perspective and stitch the images
        stitched_image = cv2.warpPerspective(images[i + 1], accumulated_homography, 
                                             (images[i].shape[1] + images[i + 1].shape[1], images[i].shape[0]))
        stitched_image[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

        # Remove the empty pixels and retain maximum image information
        # stitched_image = remove_empty_pixels(stitched_image)
        stitched_images.append(stitched_image)

    # Combine all stitched images into a final panorama
    final_panorama = stitched_images[0]
    for i in range(1, len(stitched_images)):
        final_panorama = cv2.warpPerspective(stitched_images[i], np.eye(3), 
                                             (final_panorama.shape[1] + stitched_images[i].shape[1], final_panorama.shape[0]))
        final_panorama[0:stitched_images[i].shape[0], 0:stitched_images[i].shape[1]] = stitched_images[i]

    # Crop the final image to 512x512 centered around the middle
    cropped_final_panorama = final_panorama[:512, :512]
    rotated_final_panorama = rotate_to_vertical(cropped_final_panorama)

    return rotated_final_panorama