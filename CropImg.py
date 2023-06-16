#Libraries import
import numpy as np
import cv2
import torch
import os
from PIL import Image 
from Reconnaissance_objet_images import list_images


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

imagePathsRGB = list_images("ImgRGB")
imagePathsDepth = list_images("ImgDepth")


def Cropped_images(image_paths_rgb, image_paths_depth):

    CropRGB = []
    CropDepth = []

    # Read the RGB images
    for path in image_paths_rgb:
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
                
                # Get the dimensions of the image
                height, width, channels = img_rgb.shape

                # For each image
                for l in lst:
                    
                    # Put the numpy into an image so the pixels can be changed
                    image_rgb = Image.fromarray(img_rgb)
                    image_depth = Image.fromarray(img_depth)

                    # Black banner above the object
                    for x in range(0,l[1]):
                        for y in range(0, l[4]):
                            image_rgb.putpixel((x,y),(0,0,0))
                            image_depth.putpixel((x,y),(0,0,0))

                    # Black banner on the left of the object
                    for x in range(0,l[3]):
                        for y in range(l[4],height):
                            image_rgb.putpixel((x,y),(0,0,0))
                            image_depth.putpixel((x,y),(0,0,0))

                    # Black banner below the object
                    for x in range(l[3],width):
                        for y in range(l[2], height):
                            image_rgb.putpixel((x,y),(0,0,0))
                            image_depth.putpixel((x,y),(0,0,0))

                    # Black banner on the right of the object
                    for x in range(l[1], width):
                        for y in range(0, l[2]):
                            image_rgb.putpixel((x,y),(0,0,0))
                            image_depth.putpixel((x,y),(0,0,0))

                    # Put the image back as an array
                    final_rgb = np.asarray(image_rgb)
                    final_depth = np.asarray(image_depth)

                # Save the new images in different repositories
                rgb_path = f'ImagesRGBCrop/Crop_{basename_rgb}'
                depth_path = f'ImagesDepthCrop/Crop_{basename_depth}'
                cv2.imwrite(rgb_path, final_rgb)
                cv2.imwrite(depth_path, final_depth)

            # Show the new images
            #cv2.imshow("cropped_rgb", final_rgb)
            #cv2.imshow("cropped_depth", final_depth)
            cv2.waitKey(0)

        CropRGB.append(final_rgb)
        CropDepth.append(final_depth)

    return CropRGB, CropDepth


Photos = Cropped_images(imagePathsRGB, imagePathsDepth)