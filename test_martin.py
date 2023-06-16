import mysql.connector
import cv2
import glob
import matplotlib.pyplot as plt

host ='localhost'
user='root'
password='mysqlpass'
database='labello_db_v2'

db = mysql.connector.connect(host=host,user=user,passwd=password,db=database)

cursor = db.cursor()

query="""
SELECT obj_weight,obj_id,obj_size_length_x,obj_size_width_y,obj_size_height_z FROM object WHERE obj_id;
"""
# obj_weight FROM object WHERE obj_id BETWEEN 1 AND 100;
cursor.execute(query)

results = cursor.fetchall()

liste_weight = []
liste_id_image = []
liste_length_x = []
liste_width_y = []
liste_height_z = []
for row in results:
    id = row[1]
    weight = row[0]
    length_x = row[2]
    width_y = row[3]
    height_z = row[4]
    liste_weight.append(weight)
    liste_id_image.append(id)
    liste_length_x.append(length_x)
    liste_width_y.append(width_y)
    liste_height_z.append(height_z)
cursor.close()
db.close()
print("ID:")
print(liste_id_image)
print("\n")
print("POIDS:")
print(liste_weight)
print("\n")
print("LONGUEUR:")
print(liste_length_x)
print("\n")
print("LARGEUR:")
print(liste_width_y)
print("\n")
print("HAUTEUR:")
print(liste_height_z)

# Récupération des images 

liste_path_image = []
for i in range(len(liste_weight)-1, -1, -1):
    if liste_weight[i] == 0: # Si jamais c'était une donnée de test, alors il faut virer tout les données associés : les images et les poids. ( A Reformuler )
        del liste_weight[i] # Retire de la liste le poids pour l'indice i qui est une donnée de test
        del liste_id_image[i] # Retire de la liste l'id de l'image associé à ce poid pour n'avoir plus que les données qui nous intéressent.
    else: # Sinon tout récupérer
        img_path_v1 = "../labello/dataset/img/rgb_" + str(liste_id_image[i]) + "_"
        img_path = glob.glob(img_path_v1 + "*")
        if len(img_path) > 0:
            img = cv2.imread(img_path[0])
            liste_path_image.append(img)
        else:
            print("Aucune image correspondante pour :", img_path_v1)

#print(liste_path_image)
'''for i in range(len(liste_path_image)):
    image = liste_path_image[i]
    # Afficher l'image
    plt.imshow(image)
    plt.axis('off')
    plt.show()'''



