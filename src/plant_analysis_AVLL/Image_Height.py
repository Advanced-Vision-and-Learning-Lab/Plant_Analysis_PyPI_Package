from plantcv import plantcv as pcv
import plantcv
from PIL import Image
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from skimage.filters import threshold_li


class options:
    def __init__(self, image_path):
        self.image = image_path
        self.debug = None
        self.tmp_dir = "plantpattern"
        self.result = os.path.join(self.tmp_dir, "try2.txt")
        self.writeimg = False 
        self.outdir = "plantpattern"
        os.makedirs(self.tmp_dir, exist_ok=True)

def image_height(image, camera_distance):
    
    # Get options
    args = options('')

    # Set debug to the global parameter 
    pcv.params.debug = args.debug
    original_img = image
    gray = pcv.rgb2gray(rgb_img=image)
    thresh = threshold_li(gray)
    binary = gray > thresh

    s_thresh = binary.astype(int)
    bs = pcv.logical_or(bin_img1=gray, bin_img2=np.uint8(s_thresh*255))
    masked = pcv.apply_mask(img=original_img, mask=bs, mask_color='black')
    ab_fill = pcv.fill(bin_img=s_thresh, size=10)
    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='black')
    id_objects, obj_hierarchy = pcv.find_objects(img=np.uint8(original_img*255), mask=ab_fill)
    roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=95, y=5, h=500, w=350)
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=original_img, roi_contour=roi1, 
                                                                roi_hierarchy=roi_hierarchy, 
                                                                object_contour=id_objects, 
                                                                obj_hierarchy=obj_hierarchy,
                                                                roi_type='partial')

    obj, mask = pcv.object_composition(img=original_img, contours=roi_objects, hierarchy=hierarchy3)
    analysis_image = pcv.analyze_object(img=original_img, obj=obj, mask=mask, label="default")
    boundary_image2 = pcv.analyze_bound_horizontal(img=np.uint8(original_img*255), obj=obj, mask=mask, 
                                                line_position=510, label="default")
    height_above_reference = pcv.outputs.observations['default']['height_above_reference']['value']
    #small greenhouse distance = 30cm
    #56.3 * 56.3
    image_width, image_height, h = original_img.shape
    x_offset = 56.3/512
    actual_greenhouse_height = x_offset * height_above_reference
    # print("Actual Height of plant:", actual_greenhouse_height)
    # Create a DataFrame with the actual height
    data = {'Parameter': ['Height Above Reference (pixels)', 'Plant height (cm)', 'Image Width (pixels)', 'Pixel Size (cm)'],
            'Value': [height_above_reference, actual_greenhouse_height, image_width, x_offset]}
    df = pd.DataFrame(data)

    return df, actual_greenhouse_height





