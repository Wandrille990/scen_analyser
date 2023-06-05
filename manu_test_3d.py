##afficher le nuage de point : 

        # from PIL import Image
        # from tkinter import *
        
        # # To convert the image From JPG to PNG : {Syntax}
        # img = Image.open("Image.jpg")
        # img.save("Image.png")

        # # Charger l'image 3D (par exemple, une image de profondeur)
        # image_3d = cv2.imread('image_3d.png', cv2.IMREAD_UNCHANGED)

        # # Extraire les coordonnées des points
        # points = np.argwhere(image_3d > 0)  # Points dont la valeur est supérieure à 0

        # # Récupérer les coordonnées x, y, z
        # x = points[:, 1]
        # y = points[:, 0]
        # z = image_3d[y, x]  # Valeurs de profondeur correspondantes

        # # Afficher le nuage de points
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, c=z, cmap='jet')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()
