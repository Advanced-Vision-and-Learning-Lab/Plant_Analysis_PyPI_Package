import os
import cv2
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pdb
from Connect_Components_Preprocessing import CCA_Preprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Image_Stitching import *
from Plant_Phenotypes import *
from Image_Segmentation import *
from skimage.feature import local_binary_pattern,hog
from skimage import exposure
from time import time

class Plant_Analysis:
    
    def __init__(self, device = 'cpu'):
        
        self.service_type = 0
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        self.plants = {}
        self.segmentation_model_weights_path = 'yolo_segmentation_model.pt'
        self.segmentation_model = None
        self.variable_k = 2
        self.raw_channel_names = ['Red (660 nm)', 'Green (580 nm)', 'Red Edge (730 nm)', 'NIR (820 nm)']
        self.device = device
        self.LBP_radius = 1
        self.LBP_n_points = 8*self.LBP_radius
        self.analysis_items = ['stitched_image', 'cca_image', 'segmented_image', 'tips', 'branches', 'tips_and_branches', 'sift_features', 'lbp_features', 'hog_features']
        self.statistics_items = ['height', 'width', 'area']
    
    def update_service_type(self,service):
        
        self.service_type = service
    
    def parse_folders(self):
        
        if self.service_type == 0:
            folder_path = self.input_folder_path
            plant_folders = sorted(os.listdir(folder_path))
            for plant_folder in plant_folders:
                self.plants[plant_folder] = {}
                self.plants[plant_folder]['raw_images'] = []
                plant_folder_path = os.path.join(folder_path,plant_folder)
                image_names = sorted(os.listdir(plant_folder_path))
                for image_name in image_names:
                    image_path = os.path.join(plant_folder_path,image_name)
                    image = Image.open(image_path)
                    self.plants[plant_folder]['raw_images'].append((image,image_name))
        
        if self.service_type == 1:
            plant_folder_path = self.input_folder_path
            plant_name = plant_folder_path.split('/')[-1]
            self.plants[plant_name] = {}
            self.plants[plant_folder]['raw_images'] = []
            image_names = sorted(os.listdir(plant_folder_path))
            for image_name in image_names:
                image_path = os.path.join(plant_folder_path,image_name)
                image = Image.open(image_path)
                self.plants[plant_folder]['raw_images'].append((image,image_name))
    
    def update_input_path(self,input_path):
        
        self.input_folder_path = input_path
        self.parse_folders()
        
    def update_check_RI_option(self, check_RI):
        
        self.show_raw_images = check_RI

    def update_check_CI_option(self, check_CI):
        
        self.show_color_images = check_CI
    
    def load_segmentation_model(self):
        
        self.segmentation_model = load_yolo_model(self.segmentation_model_weights_path)

    def get_plant_names(self):

        return sorted(list(self.plants.keys()))

    def get_raw_images(self, plant):

        # return [(image.astype(np.uint8), image_name) for image,image_name in self.plants[plant]['raw_images']]
        return self.plants[plant]['raw_images']
        
    def get_color_images(self, plant):

        return [(image.astype(np.uint8), image_name) for image,image_name in self.plants[plant]['color_images']]

    def get_plant_analysis_images(self, plant):

        return [(image.astype(np.uint8), image_name) for image,image_name in [self.plants[plant][item] for item in self.analysis_items]]

    def get_plant_height(self, plant):

        return str(round(self.plants[plant]['height'],2))+' cm'
    
    def get_plant_statistics_df_plantwise(self, plant):
        
        return pd.DataFrame({'Phenotype': ['Height', 'Width', 'Area'], 'Value': [str(round(self.plants[plant]['height'],2))+' cm', str(round(self.plants[plant]['width'],2))+' cm', str(round(self.plants[plant]['area'],2))+' square cm']})
        
    def tile(self, image, d=2):
        
        w, h = image.size
        grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
        boxes = []
        
        for i, j in grid:
            box = (j, i, j+d, i+d)
            boxes.append(box)

        return boxes

        
    def make_color_images(self):
        
        for plant_name in self.plants.keys():
            
            self.plants[plant_name]['color_images'] = []
            
            for raw_image, image_name in self.plants[plant_name]['raw_images']:
                
                size = raw_image.size[0] // 2
                slices = self.tile(raw_image, d = size)
                index = 0                
                image_stack = np.zeros((size, size, len(slices)))
                
                for box in slices:
                    
                    image_stack[:, :, index] = np.array(raw_image.crop(box))
                    index += 1
                    
                red = np.expand_dims(image_stack[:, :, 1], axis=-1)
                green = np.expand_dims(image_stack[:, :, 0], axis=-1)
                red_edge = np.expand_dims(image_stack[:, :, 2], axis=-1)
                NIR = np.expand_dims(image_stack[:, :, -1], axis=-1)
                
                composite_image = np.concatenate((green, red_edge, red), axis=-1) * 255
                normalized_image = ((composite_image - composite_image.min())*255 / (composite_image.max() - composite_image.min())).astype(np.uint8)
                
                self.plants[plant_name]['color_images'].append((normalized_image, image_name))
                
    def stitch_color_images(self):
        
        for plant_name in self.plants.keys():
            
            input_images = [color_image for color_image,image_name in self.plants[plant_name]['color_images']]
            stitched_image = image_stitching(input_images)
            self.plants[plant_name]['stitched_image'] = (stitched_image, 'Whole Plant Image')
            
    def calculate_connected_components(self):
        
        for plant_name in self.plants.keys():
            
            gray_image, binary = CCA_Preprocess(self.plants[plant_name]['stitched_image'][0], k = self.variable_k)
            preprocessed_image = np.repeat(np.expand_dims(binary, axis=-1), 3, axis=-1) * self.plants[plant_name]['stitched_image'][0]
            cca_image = 255*(preprocessed_image - preprocessed_image.min()) / (preprocessed_image.max() - preprocessed_image.min())
            cca_image = cca_image.astype(np.uint8)
            self.plants[plant_name]['cca_image'] = (cca_image, 'Background Separated Using Connected Component Analysis')
        
    def run_segmentation(self):
        
        input_images, plant_names = [self.plants[plant_name]['stitched_image'][0] for plant_name in sorted(list(self.plants.keys()))], [plant_name for plant_name in sorted(list(self.plants.keys()))]
        results = self.segmentation_model.predict(input_images, conf = 0.128, device = self.device)
        
        for result_index in range(len(results)):
            
            result = results[result_index]
            
            if result:
            
                mask = preprocess_mask(result.masks.data)
                binary_mask_np = generate_binary_mask(mask)
                overlayed_image = overlay_mask_on_image(binary_mask_np, self.plants[plant_names[result_index]]['stitched_image'][0])
                self.plants[plant_names[result_index]]['segmented_image'] = (overlayed_image, 'Background Separated Using Image Segmentation')
    
    def calculate_plant_phenotypes(self):

        for plant_name in sorted(list(self.plants.keys())):

            phenotypes = get_plant_phenotypes(self.plants[plant_name]['segmented_image'][0], offset = 56.3/512)
            self.plants[plant_name]['height'] = phenotypes['Plant Height (cm)']
            self.plants[plant_name]['width'] = phenotypes['Plant Width (cm)']
            self.plants[plant_name]['area'] = phenotypes['Plant Area (square cm)']
        
    def calculate_tips_and_branches(self):

        for plant_name in sorted(list(self.plants.keys())):

            # pcv.outputs.clear()
            gray_image = cv2.cvtColor(self.plants[plant_name]['segmented_image'][0], cv2.COLOR_RGB2GRAY)
            skeleton = pcv.morphology.skeletonize(mask = gray_image)
            tips = pcv.morphology.find_tips(skel_img = skeleton, mask = None, label = plant_name)
            branches = pcv.morphology.find_branch_pts(skel_img = skeleton, mask = None, label = plant_name)
            tips_and_branches = np.zeros_like(skeleton)
            tips_and_branches[tips > 0] = 255
            tips_and_branches[branches > 0] = 128
            kernel = np.ones((5, 5), np.uint8)
            tips = cv2.dilate(tips, kernel, iterations = 1)
            branches = cv2.dilate(branches, kernel, iterations = 1)
            tips_and_branches = cv2.dilate(tips_and_branches, kernel, iterations = 1)
            self.plants[plant_name]['tips'] = (tips, 'Plant Tips')
            self.plants[plant_name]['branches'] = (branches, 'Plant Branch Points')
            self.plants[plant_name]['tips_and_branches'] = (tips_and_branches, 'Plant Tips and Branch Points')
            self.plants[plant_name]['gray_image'] = (gray_image, 'Gray Segmented Image')
            self.plants[plant_name]['skeleton'] = (skeleton, 'Morphology Skeleton')
        
    def calculate_sift_features(self):

        for plant_name in sorted(list(self.plants.keys())):

            sift = cv2.SIFT_create()
            kp, des= sift.detectAndCompute(self.plants[plant_name]['skeleton'][0], None)
            sift_image = cv2.drawKeypoints(self.plants[plant_name]['skeleton'][0], kp, des)
            self.plants[plant_name]['sift_features'] = (sift_image, 'SIFT Features')
        
    def calculate_LBP_features(self):

        for plant_name in sorted(list(self.plants.keys())):

            lbp = local_binary_pattern(self.plants[plant_name]['gray_image'][0], self.LBP_n_points, self.LBP_radius)
            self.plants[plant_name]['lbp_features'] = (lbp, 'Local Binary Patterns')
        
    def calculate_HOG_features(self):

        for plant_name in sorted(list(self.plants.keys())):

            fd,hog_image = hog(self.plants[plant_name]['gray_image'][0], orientations=10, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False, channel_axis=-1)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_image_rescaled = hog_image_rescaled*255
            self.plants[plant_name]['hog_features'] = (hog_image_rescaled, 'Histogram of Oriented Gradients')

    def reset(self):

        self.service_type = 0
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        del self.plants
        self.plants = {}

    def get_plant_statistics_df(self):

        plants_names = sorted(list(self.plants.keys()))
        heights = []
        widths = []
        areas = []
        
        for plant_name in plants_names:
            
            heights.append(round(self.plants[plant_name]['height'],2))
            widths.append(round(self.plants[plant_name]['width'],2))
            areas.append(round(self.plants[plant_name]['area'],2))

        return pd.DataFrame({'Plant_Name':plants_names, 'Height': heights, 'Width': widths, 'Area': areas})

    def make_dir(self, folder):

        if not os.path.exists(folder):

            os.mkdir(folder)
    
    def save_results(self, folder_path):

        self.make_dir(folder_path)
        
        filepath = os.path.join(folder_path, 'plants_features_and_statistics.txt')
        f = open(filepath, 'w')
        outputs = pcv.outputs.observations
        
        for plant_name in sorted(list(self.plants.keys())):
        
            tips_list = outputs[plant_name]['tips']['value']
            branch_pts_list = outputs[plant_name]['branch_pts']['value']
            line = plant_name+',tips,'+','.join([str(coord[0])+','+str(coord[1]) for coord in tips_list])+'\n'
            f.write(line)
            line = plant_name+',branch_points,'+','.join([str(coord[0])+','+str(coord[1]) for coord in branch_pts_list])+'\n'
            f.write(line)

            for item in self.statistics_items:
                
                line = plant_name+','+item+','+str(self.plants[plant_name][item])+'\n'
                f.write(line)
        
        f.close()
        
        for plant_name in sorted(list(self.plants.keys())):

            plant_folder = os.path.join(folder_path, plant_name)
            self.make_dir(plant_folder)
            color_images_folder = os.path.join(plant_folder, 'Color_Images')
            self.make_dir(color_images_folder)
            
            for image, image_name in self.plants[plant_name]['color_images']:

                cv2.imwrite(os.path.join(color_images_folder,image_name),image)
                
            for item in self.analysis_items:

                image,name = self.plants[plant_name][item]
                cv2.imwrite(os.path.join(plant_folder,'_'.join(name.split(' '))+'.jpg'), image)
