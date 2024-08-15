# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:09:11 2022

@author: jpeeples
"""
import cv2
from skimage.filters import threshold_li
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def CCA_Preprocess(composite_img, k=2):

    #Use pca to reduce vector 
    reshaped_composite_img = np.reshape(composite_img,(-1,3))
    
    #Apply PCA
    pca = PCA(n_components=1, whiten=True)
    gray_vector = pca.fit_transform(reshaped_composite_img)
    
    #Visualize image
    gray_img = np.reshape(gray_vector,(512,512))
    
    #Normalize and scale between 0 and 255 (inclusive)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(gray_img)
    gray_img = scaler.transform(gray_img)
    
    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(np.uint8(gray_img*255), (5, 5), 0)
    
    #Sharpen Image
    kernel3 = np.array([[0, -1,  0],
                       [-1,  5, -1],
                        [0, -1,  0]])
    
    blurred = cv2.filter2D(src=blurred, ddepth=-1, kernel=kernel3)
    
    
    # #Threshold image from background for CCA
    thresh = threshold_li(gray_img)
    binary = gray_img > thresh

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 400))

    # Mask = cv2.erode(np.uint8(binary*255), kernel)

    # # Expand the mask in the vertical direction
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 400))
    # Mask = cv2.dilate(Mask, kernel)

    # # Erase the connection by placing zeros
    # binary[np.where(Mask != 0)] = 0

    
    img = np.repeat(np.expand_dims(binary * gray_img, axis = -1), 3, axis=-1)
     
    # Applying threshold
    threshold = np.uint8(binary*255)
     
    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
     
    # Initialize a new image to
    # store all the output components
    output = np.zeros(gray_img.shape, dtype="uint8")
     
    # Loop through each component
    for i in range(1, totalLabels):
       
        # if (area > 10000) and (area < 50000):
        if i in np.argsort(values[:,4])[-k:]:
            # Create a new image for bounding boxes
            new_img=img.copy()
             
            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
             
            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1+ w, y1+ h)
            (X, Y) = centroid[i]
             
            # Bounding boxes for each component
            cv2.rectangle(new_img,pt1,pt2,
                          (0, 255, 0), 3)
            cv2.circle(new_img, (int(X),
                                 int(Y)),
                       4, (0, 0, 255), -1)
     
            # Create a new array to show individual component
            component = np.zeros(gray_img.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255
     
            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component,componentMask)
            output = cv2.bitwise_or(output, componentMask)
             
#     fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
#     ax = axes.ravel()
#     ax[0] = plt.subplot(1, 3, 1)
#     ax[1] = plt.subplot(1, 3, 2, sharex=ax[0], sharey=ax[0])
#     ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
#     # ax[3] = plt.subplot(1,4,4,sharex=ax[0], sharey=ax[0])
    
#     ax[0].imshow(gray_img, cmap='BuGn')
#     ax[0].set_title('Original')
#     ax[0].axis('off')
    
#     ax[1].imshow(output, cmap='BuGn')
#     ax[1].set_title('Connected Components')
#     ax[1].axis('off')
    
#     ax[2].imshow(gray_img*(output/255), cmap='BuGn')
#     ax[2].set_title('ROI')
#     ax[2].axis('off')
    
#     plt.suptitle(title)
    
#     plt.tight_layout()
#     fig.savefig('{}/Img_{}.png'.format(folder,img_index))
#     plt.close()
    
    #Return gray image and mask
    return gray_img, output/255