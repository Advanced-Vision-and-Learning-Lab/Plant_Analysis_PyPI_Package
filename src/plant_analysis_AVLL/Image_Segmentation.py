import os
HOME = os.getcwd()
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.filters import threshold_otsu
from PIL import Image
import pdb
import torchvision.transforms as T

def load_yolo_model(weights_path, image_size=(512, 512)):
    model = YOLO(weights_path)
    model.img_size = image_size
    return model

def detect_object(model, image_path, confidence=0.128, device='cpu'):
    results = list(model.predict(image_path, conf=confidence, device = device))
    return results[0] if results else None

def preprocess_mask(mask, target_size=(512, 512)):
    mask_pil = T.ToPILImage()(mask)
    mask_np = np.array(mask_pil)
    resized_mask = resize(mask_np, target_size, mode='constant')
    return resized_mask

def generate_binary_mask(mask, target_size=(512, 512)):
    resized_mask = cv2.resize(mask, target_size)
    # Check if the mask has two channels
    print(resized_mask.shape)
    if len(resized_mask.shape) == 3:
        # Convert the mask to grayscale
        gray_mask = resized_mask[:, :, 0]
    else:
        gray_mask = resized_mask
    
    _, binary_mask = cv2.threshold(np.uint8(gray_mask * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask_np = np.array(binary_mask) / 255
    return binary_mask_np

def overlay_mask_on_image(binary_mask, image):
    image_np = np.array(image)
    binary_mask_rgb = binary_mask.astype(image_np.dtype)
    binary_mask_rgb = np.repeat(binary_mask_rgb[:, :, np.newaxis], 3, axis=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlayed_image = binary_mask_rgb * image_rgb
    return overlayed_image

def save_overlayed_image(image_array, output_path):
    image_array.save(output_path)
