#Libraries import
import numpy as np
import cv2
import torch
import os

def list_images(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    imagePaths = []
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                imagePaths.append(imagePath)
    
    return imagePaths


imagePathsRGB = list_images("ImagesRGB")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom


def Object_recognition(image_paths_rgb):

    Recog_RGB = []

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

        # Recognize the object on the image
            for l in lst:
                cv2.rectangle(img_rgb,(l[1],l[2]),(l[3],l[4]),(0,255,0),1)
                cv2.putText(img_rgb, l[0], (l[1],l[2]), 0, 0.3, (0, 255, 0))

        # Save the new image with the recognized object
            rgb_path = f'ImagesRGBRecog/Recog_{basename_rgb}'
            cv2.imwrite(rgb_path, img_rgb)

            # Show the image with the recognized object
            #cv2.imshow("Recog", img_rgb)
            cv2.waitKey(0)

        Recog_RGB.append(img_rgb)

    return Recog_RGB


Photos = Object_recognition(imagePathsRGB)