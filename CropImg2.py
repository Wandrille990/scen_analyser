#Libraries import
import numpy as np
import cv2
import torch
import os
from Reconnaissance_objet_images import list_images


imagePathsRGB = list_images("ImgRGB")
imagePathsDepth = list_images("ImgDepth")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom


def Cropped_images_2(image_paths_rgb, image_paths_depth):

    CropRGB = []
    CropDepth = []

    for path in image_paths_rgb:
        #Read all RGB images
        img_rgb = cv2.imread(path)
        basename_rgb = os.path.basename(path)
        
        # Keep only not empty images
        image = np.reshape(img_rgb, (-1,1))
        u, count_unique = np.unique(image, return_counts = True)
        if count_unique.size < 10:
            return "Image is empty"
        else:
            results = model(img_rgb)
            data = results.pandas().xyxy[0]

        # Get the coordinates and the name of the object on the image
            lst = []
            for index, row in data.iterrows():
                line = []
                line.append(row["name"])
                line.append(int(row["xmin"]))
                line.append(int(row["ymin"]))
                line.append(int(row["xmax"]))
                line.append(int(row["ymax"]))
                lst.append(line)

            # Read the depth images
            for path_depth in image_paths_depth:
                img_depth = cv2.imread(path_depth)
                basename_depth = os.path.basename(path_depth)

                # Crop the object on the image to get a new one
                for l in lst:
                    crop_img_rgb = img_rgb[l[2]:l[4], l[1]:l[3]]
                    crop_img_depth = img_depth[l[2]:l[4], l[1]:l[3]]

                # Save the new images in different repositories
                rgb_path = f'ImagesRGBCrop/Crop_{basename_rgb}'
                depth_path = f'ImagesDepthCrop/Crop_{basename_depth}'
                cv2.imwrite(rgb_path, crop_img_rgb)
                cv2.imwrite(depth_path, crop_img_depth)

            # Show the new images
            #cv2.imshow("cropped", crop_img_rgb)
            #cv2.imshow("cropped_depth", crop_img_depth)
            cv2.waitKey(0)

        CropRGB.append(crop_img_rgb)
        CropDepth.append(crop_img_depth)

    return CropRGB, CropDepth


Photos = Cropped_images_2(imagePathsRGB, imagePathsDepth)