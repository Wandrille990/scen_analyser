# Récupération des images

# Listes récupérées à partir de la base de donnée :
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf
print("Début")
liste_weight = [0, 376, 376, 188, 147, 350, 792, 503, 353, 342, 272, 645, 628, 781,
                1742, 0, 1945, 583, 567, 412, 1044, 549, 655, 571, 116]
''', 665, 372, 175,
                148, 349, 7, 532, 167, 383, 377, 1836, 220, 153, 117, 168, 296, 188, 139, 139,
                136, 320, 0, 164, 100, 0, 0, 0, 0, 0, 0, 20, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                134, 69, 192, 397, 960, 960, 90, 35, 556, 160, 129, 38, 243, 1108, 1294, 25, 20, 22,
                26, 32, 0, 35, 83, 62, 48, 16, 105, 31, 23, 115, 187, 200, 151, 28, 217, 24, 34, 0,
                7, 85, 107, 321, 57, 211, 99, 51, 327, 635, 635, 62, 17, 23, 26, 21, 24, 25, 25, 29,
                245, 199, 330, 543, 222, 388, 368, 302, 348, 45, 456, 321, 337, 420, 17, 23, 430, 379,
                63, 674, 841, 82, 0, 196, 277, 759, 44, 225, 173, 240, 371, 311, 308, 1612, 1042, 0, 86,
                1165, 316, 607, 892, 871, 48, 22, 48, 13, 49, 70, 60, 721, 301, 900, 540, 590, 630, 0,
                 447, 500, 780, 1200, 338, 460, 380, 18, 23, 74, 76, 71, 19, 31, 83, 131]'''


liste_id_image =[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26]
''', 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
                95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
                186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]'''

liste_weight_repetes = []
liste_path_image = []
for i in range(len(liste_weight)-1, -1, -1):
    if liste_weight[i] == 0: # Si jamais c'était une donnée de test, alors il faut virer toutes les données associées : les images et les poids. ( À reformuler )
        del liste_weight[i] # Retire de la liste le poids pour l'indice i qui est une donnée de test
        del liste_id_image[i] # Retire de la liste l'id de l'image associé à ce poids pour n'avoir plus que les données qui nous intéressent.
    else: # Sinon tout récupérer
        img_path_v1 = "../labello/dataset/img/rgb_" + str(liste_id_image[i]) + "_"
        img_paths = glob.glob(img_path_v1 + "*") # Liste de tous les chemins d'images correspondantes
        if len(img_paths) > 0:
            for img_path in img_paths: #récupère les 12 photos du même objet
                img = cv2.imread(img_path)
                liste_path_image.append(img)
                # Duplicate the weight for each image
                liste_weight_repetes.append(liste_weight[i])

        else:
            print("Aucune image correspondante pour :", img_path_v1)

#print(liste_path_image)
liste_weight_repetes.reverse()
print(len(liste_weight_repetes))
print(len(liste_path_image))
#for image in liste_path_image:
#for i in range(len(liste_path_image)):
    #image = liste_path_image[i]
    # Afficher l'image
    #plt.imshow(image)
    #plt.axis('off')
    #plt.show()

image_width = 640
image_height = 480

# Création du modèle
# Définition du modèle
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

# Création du modèle v2 plus performant et adapté à priori.
'''model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))'''
# Compilation du modèle.
model.compile(optimizer=Adam(), loss='mean_squared_error')

##Transformation en tableaux numpy pour que ce soit dans un format adapté au modèle
train_images = np.array(liste_path_image)
train_images = np.reshape(train_images, (len(train_images), image_width, image_height, 3))

train_weight = np.array(liste_weight_repetes)
train_weight = np.reshape(train_weight, (len(train_weight), 1))

# Séparation données d'entraînement et données de tests.
train_images, test_images, train_weight, test_weight = train_test_split(liste_path_image, liste_weight_repetes, test_size=0.3, shuffle=True, random_state=42)

train_images = np.array(train_images)
train_images = np.reshape(train_images, (len(train_images), image_width, image_height, 3))

train_weight = np.array(train_weight)

test_images = np.array(test_images)
test_images = np.reshape(test_images, (len(test_images), image_width, image_height, 3))

test_weight = np.array(test_weight)

# Entraînement du modèle.
model.fit(train_images, train_weight, epochs=2, batch_size=32)

predicted_weight = model.predict(test_images)
print("Poids prédit :", predicted_weight)