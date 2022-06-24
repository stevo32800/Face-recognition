# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:38:02 2022

@author: steve
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, shutil
import re



# separation des images en 3 pour le test, le train et la validation:
    
original_data = '/Users/steve/Documents/DL/stanford_Dogs/dog'  
  
base_dir = '/Users/steve/Documents/DL/stanford_Dogs/portion_data'

if not os.path.exists(base_dir):
    os.mkdir(base_dir) 
#creation des dossiers
#train:
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
#test:
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir) 
#val:
val_dir = os.path.join(base_dir, 'val')
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
    
    
#creation boucle pour creer des dossiers en fonction des classes, iterer sur le dossier dog et recuperer le nom de chaque classe
# le faire pour la validation, le train et le test

#creation de la liste des differentes races
my_list = []
for i in os.listdir(original_data):
    race_name = re.sub('^(.*-)',"", i)
    my_list.append(race_name)
    #print(race_name)
print(my_list)
print(len(my_list))


#creation des races dans le train
list_dir_train = []
for i in my_list:
    train_dog = os.path.join(train_dir, i)
    list_dir_train.append(train_dog)
    if not os.path.exists(train_dog):
        os.mkdir(train_dog)
print(list_dir_train)
        
    

#creation des races dans le test
list_dir_test = []
for i in my_list:
    test_dog = os.path.join(test_dir, i)
    list_dir_test.append(train_dog)
    if not os.path.exists(test_dog):
        os.mkdir(train_dog)
print(list_dir_test)
#creation des races dans le val
list_dir_val =[]
for i in my_list:
    val_dog = os.path.join(val_dir, i)
    list_dir_val.append(train_dog)
    if not os.path.exists(val_dog):
        os.mkdir(train_dog)
print(list_dir_val)


#to check if the path of the folder works i succeded to open it with python
#import webbrowser
#path = list_dir_val[0]
#webbrowser.open(path) # Opens 'PycharmProjects' folder.

      
































