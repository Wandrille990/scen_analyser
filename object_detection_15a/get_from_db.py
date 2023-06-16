import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import numpy as np


engine = db.create_engine("mysql://root:mysqlpass@127.0.0.1/labello_db_v2", isolation_level="READ UNCOMMITTED")
connection = engine.connect()


# def get_id_image(id):
#     statement = text("SELECT img_id FROM image WHERE obj_id = " + str(id))
#     id_image_results_as_dict = connection.execute(statement)
#     id_image = id_image_results_as_dict.mappings().all()[0]['img_id']
#     return id_image 

def get_weight(id):
    statement = text("SELECT obj_weight FROM object WHERE obj_id = " + str(id))
    weight_results_as_dict = connection.execute(statement)
    weight = weight_results_as_dict.mappings().all()[0]['obj_weight']
    return weight


def get_obj_wordnet_word(id):
    statement = text("SELECT obj_wordnet_word FROM object WHERE obj_id = " + str(id))
    wordnet_word_results_as_dict = connection.execute(statement)
    wordnet_word = wordnet_word_results_as_dict.mappings().all()[0]['obj_wordnet_word']
    return wordnet_word

def get_length(id):
    statement = text("SELECT obj_size_length_x FROM object WHERE obj_id = " + str(id))
    length_results_as_dict = connection.execute(statement)
    length = length_results_as_dict.mappings().all()[0]['obj_size_length_x']
    return length

def get_width(id):
    statement = text("SELECT obj_size_width_y FROM object WHERE obj_id = " + str(id))
    width_results_as_dict = connection.execute(statement)
    width = width_results_as_dict.mappings().all()[0]['obj_size_width_y']
    return width

def get_height(id):
    statement = text("SELECT obj_size_height_z FROM object WHERE obj_id = " + str(id))
    height_results_as_dict = connection.execute(statement)
    height = height_results_as_dict.mappings().all()[0]['obj_size_height_z']
    return height

def get_images_names_from_obj_id(id):
    statement = text("SELECT img_rgb_name FROM image WHERE img_obj_id = " + str(id))
    images_names_results_as_dict = connection.execute(statement)
    images_names = list(list(elem.values())[0] for elem in images_names_results_as_dict.mappings().all())
    return images_names 

if __name__ == '__main__':

    #id_obj = 200 #donner le chemin d'accès aux images puis commandes python pour garder l'id entre les deux _

    #weight_test = get_weight(id_image)
    #length_test = get_length(id_image)
    #width_test = get_width(id_image)
    #height_test = get_height(id_image)
    #wordnet_word = get_obj_wordnet_word(id_image)
    #images_names = get_images_names_from_obj_id(id_obj)

    #print(images_names)
    #print(len(images_names))

    #print(wordnet_word, weight_test,length_test, width_test, height_test)
    pass
