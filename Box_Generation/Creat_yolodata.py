import argparse
import glob,shutil, os,random
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import os




def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["p", "pm,"     help="Which downstream task.")
    args = parser.parse_args()


################################################################################################################################
    if  args.dataset == "p" :

        all_images = glob.glob('/root/autodl-tmp/p-split_dataset/box/test/images/*.jpg')
        gt_pathhh= os.path.join('/root/autodl-tmp/p-split_dataset/box/test/masks')
        output_dir_label = "/root/autodl-tmp/sun_seg/new_p_box-valid/labels"
        output_dir_image = "/root/autodl-tmp/sun_seg/new_p_box-valid/images"
        
        seg_value = 1.
      
        for i in range(len(all_images)):
            #gt_path = all_images[i].split('/')
            #gt_path[4]='GT'
            #image_name = gt_path[-1].split('.')[0]
            #gt_path[-1] = image_name + '.png'
            #gt_path = '/'.join(gt_path)
            gt_path = os.path.join(gt_pathhh,  format(i+1) + '.jpg')
            print( gt_path)
            mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)/255
            segmentation = np.where(mask_np == seg_value)
            heigth,width = mask_np.shape
            
            # Bounding Box
            bboxes = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
            
                bboxes = x_min, y_min, x_max, y_max
            
            X_C = (x_max+x_min)/2/width
            Y_C = (y_max+y_min)/2/heigth 
            W = (x_max - x_min)/width
            H = (y_max-y_min)/heigth 
            df = pd.DataFrame([[0, X_C, Y_C, W, H]])
            
            output_file_path = os.path.join(output_dir_label, format(i+1)+ '.txt')
            df.to_csv(output_file_path, sep='\t', index=False, header=False)
            shutil.copy(all_images[i], output_dir_image)
        
        
        


################################################################################################################################
    else:
        print("dataset not supported")


if __name__ == "__main__":
    main()