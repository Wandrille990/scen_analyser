import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import mysql.connector


# Établir la connexion à la base de données
conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="mysqlpass",
  database="labello_db_v2"
)

# Créer un curseur pour exécuter les requêtes SQL
cursor = conn.cursor()

# Exécuter la requête SQL pour récupérer les ID
requet = "SELECT DISTINCT obj_id, obj_weight,obj_size_length_x, obj_size_width_y, obj_size_height_z, obj_wordnet_word FROM object WHERE obj_id BETWEEN 71 AND 295 ORDER BY obj_id ASC"
cursor.execute(requet)

results = cursor.fetchall()
data = {
    'Id': [],
    'weight_object': [],
    'length_x_object':[],
    'width_y_object':[],
    'height_z_object':[],
    'obj_wordnet_word' : [],
}

# Parcourir les résultats et ajouter les données au tableau
for row in results:
    if not (('NOT' in row[5]) or (row[0]==76) or ('ball' in row[5])): # 
        data['Id'].append(row[0])
        data['weight_object'].append(row[1])
        data['length_x_object'].append(row[2])
        data['width_y_object'].append(row[3])
        data['height_z_object'].append(row[4])
        data['obj_wordnet_word'].append(row[5])

selected_row_data = pd.DataFrame(data)
#print(selected_row_data.info())
#print(selected_row_data.head())

#search = selected_row_data.loc[(selected_row_data['weight_object'] > 10)]
#print(search)

selected_row_data.to_csv('/home/user/Documents/selected_row_data.csv', index=False)
one_hot_encoder = OneHotEncoder(sparse=False)
selected_encoded_data = one_hot_encoder.fit_transform(selected_row_data[['obj_wordnet_word']])
selected_encoded_type_categorie = one_hot_encoder.get_feature_names_out(['obj_wordnet_word'])

with open("list_order_onehotencoded.csv", 'w') as f:
  for cat in selected_encoded_type_categorie:
    f.write(cat+",")

#with open("list_order_onehotencoded.csv", 'r') as f:
#    pass

print(selected_encoded_type_categorie)


# Créer le DataFrame encodé avec les noms de colonnes appropriés
onehot_encoded_columns = pd.DataFrame(selected_encoded_data, columns=selected_encoded_type_categorie)
#print(onehot_encoded_columns.head())


################################

# # Fusionner le DataFrame encodé avec le DataFrame existant

selected_row_data_and_onehot_encoded_columns = pd.concat([selected_row_data, onehot_encoded_columns], axis=1)

selected_row_data_and_onehot_encoded_columns.to_csv('/home/user/Documents/selected_row_data_and_onehot_encoded_columns.csv', index=False)
# # # Affichage du DataFrame
#print(selected_row_data_and_onehot_encoded_columns.info())
#print(selected_row_data_and_onehot_encoded_columns.head())



selected_row_data_and_onehot_encoded_columns = pd.read_csv('/home/user/Documents/selected_row_data_and_onehot_encoded_columns.csv')
X = selected_row_data_and_onehot_encoded_columns.drop(['Id', 'weight_object', 'obj_wordnet_word'], axis=1)
#on met dans la variable X toutes les colonnes excepté la dernière (les variables explicatives)
#Y = Data.iloc[:,1] #on met dans Y seulement la dernière colonne (la variable cible)
Y = selected_row_data_and_onehot_encoded_columns[['weight_object']]

#print(selected_row_data_and_onehot_encoded_columns.isnull().any().sum())
#print(selected_row_data_and_onehot_encoded_columns.head())

#sns.heatmap(selected_row_data_and_onehot_encoded_columns.drop(['Id','obj_wordnet_word'], axis=1).corr(), annot=True, vmax=1, vmin=-1)
#plt.show()

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2) #utilisation de 20% de la base pour les test

#Construction du modèle
regressor = LinearRegression()
#regressor.fit(X_train,Y_train)
regressor.fit(X,Y)

#Faire des prédictions
#score = regressor.score(X_test, Y_test)
score = regressor.score(X, Y)
print("Score R² du modèle :", score)


id = 92
requet = "SELECT obj_id, obj_weight, obj_size_length_x, obj_size_width_y, obj_size_height_z, obj_wordnet_word FROM object WHERE obj_id = " + str(id)
cursor.execute(requet)
results = cursor.fetchall()
#print(results)

to_pandas = {
    'Id': [],
    'weight_object': [],
    'length_x_object':[],
    'width_y_object':[],
    'height_z_object':[],
    'obj_wordnet_word' : [],
}

for row in results:
    to_pandas['Id'].append(row[0])
    to_pandas['weight_object'].append(row[1])
    to_pandas['length_x_object'].append(row[2])
    to_pandas['width_y_object'].append(row[3])
    to_pandas['height_z_object'].append(row[4])
    to_pandas['obj_wordnet_word'].append(row[5])

object_to_predict = pd.DataFrame(to_pandas)
#print(object_to_predict.head())


encoded_object_to_predict = one_hot_encoder.transform(object_to_predict[['obj_wordnet_word']])
onehot_encoded_object_to_predict_columns = pd.DataFrame(encoded_object_to_predict, columns=selected_encoded_type_categorie)
object_to_predict = pd.concat([object_to_predict, onehot_encoded_object_to_predict_columns], axis=1)
#print(object_to_predict.head())


#X = object_to_predict.drop(['Id', 'weight_object', 'obj_wordnet_word'], axis=1)

Y_pred = regressor.predict(X)
#print(Y_pred)

MSE_regressor = mean_squared_error(Y, Y_pred)
print("MSE du regresseur :", MSE_regressor)


#sns.boxplot(selected_row_data['weight_object'])
#plt.show()

#resultat = selected_row_data.loc[selected_row_data['weight_object'] > 2500, 'Id']
#print(resultat)

def pourcentage(Y_true, total):
    for
    pourcentage = (Y_true / total) * 100
    return pourcentage
# Fermer le curseur et la connexion à la base de données
cursor.close()
conn.close()

















