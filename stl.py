import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy import ndimage, misc
import skimage
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import streamlit as st
from skimage import io
from keras.preprocessing.image import img_to_array
import cv2
import facial_app1 as fa
import os
from keras.models import load_model
import os.path
from os import path
import time
from random import randint
from time import sleep
import pickle

#model = load_model('my_model.h5')   
 
#namevid_val = ['celia_val','silvain_val','steve_val']





st.title("TUTO FACE DETECTOR")

# st.write("""
# # Explore different classifier
         
#  """)

import PIL 

from PIL import Image
image = Image.open('face_app.jpg')

st.image(image, caption='Super app')



mode = st.sidebar.radio(
     "What do you want to do ? ",
     ('Add people', 'Make the model','Mode test',))


if mode == 'Add people':
     st.write('You selected Add people')
     number_p =  st.sidebar.number_input('Insert the number of people', step=1,min_value=1, max_value=10 )
     list_p = []
     if number_p:
         if not path.exists('image_train_essaie'):            
             os.mkdir('image_train_essaie')
         if not path.exists('image_val_essaie'):   
             os.mkdir('image_val_essaie')
             
         path_title_list_train = []
         path_title_list_val = []
         list_p = []
         list_v_train = []         
         list_v_val = []
         for i in range(0, number_p):
             st.write('debut de la boucle',i+1)
             title = st.text_input('name of the person', '', key = i)
             list_p.append(title)
             
             parent_dir_train = '/Users/steve/Documents/DL/image_train_essaie'             
             path_title_train = os.path.join(parent_dir_train, list_p[i])
             #print(path_title_train)
             #st.write(path_title_train)
             path_title_list_train.append(path_title_train)
             
             
             parent_dir_val = '/Users/steve/Documents/DL/image_val_essaie'
             path_title_val = os.path.join(parent_dir_val, list_p[i])
             path_title_list_val.append(path_title_val)
             #st.write(path_title_val)
             
             
             #st.write(path_title_list_train[i])
             if not os.path.exists(path_title_list_train[i]):                 
                  os.mkdir(path_title_list_train[i])
                  #st.write('creation du dossier')
                  #st.write(i)
             #else:
                  #st.write('le dossier exist deja')
                  #st.write(i)
                  #st.write(path_title_list_train[i])
             if not os.path.exists(path_title_list_val[i]):                 
                  os.mkdir(path_title_list_val[i])
                  #st.write('creation du dossier')
             #else:
                  #st.write('le dossier exist deja')
             
             
             ###########train video#############
             uploaded_video_train = st.file_uploader("Choose a video for the train set",accept_multiple_files=True,
                                                  key = i, type=["mp4", "mov"])
             
             if uploaded_video_train:
                
                 for uploaded_file in uploaded_video_train:
                     uploaded_video_n_train =  uploaded_file.name                     
                     uploaded_video_t_train =  uploaded_file.type
                     uploaded_video_s_train =  uploaded_file.size
                     uploaded_video_train = uploaded_file
                
                 file_details_train = {"filename":uploaded_video_n_train, "filetype":uploaded_video_t_train,
                              "filesize":uploaded_video_s_train}
                 #st.write(file_details_train)
                 global path_train_essaie
                 path_train_essaie = 'image_train_essaie'
                 with open(os.path.join(f'{path_train_essaie}',uploaded_video_n_train),"wb") as f:
                   f.write((uploaded_video_train).getbuffer())
                   st.success("File Saved")
                 list_v_train.append(uploaded_video_n_train)   
                   #####validation video#####
             uploaded_video_val = st.file_uploader("Choose a video for the validation set",accept_multiple_files=True,
                                                  key = i+1, type=["mp4", "mov"])
             
             if uploaded_video_val:
                 
                 for uploaded_file in uploaded_video_val:
                     uploaded_video_n_val =  uploaded_file.name
                     
                     uploaded_video_t_val =  uploaded_file.type
                     uploaded_video_s_val =  uploaded_file.size
                     uploaded_video_val = uploaded_file
                 list_v_val.append(uploaded_video_n_val)
                 file_details_val = {"filename":uploaded_video_n_val, "filetype":uploaded_video_t_val,
                              "filesize":uploaded_video_s_val}
                 #st.write(file_details_val)
                 global path_val_essaie
                 path_val_essaie = 'image_val_essaie'
                 with open(os.path.join(f'{path_val_essaie}',uploaded_video_n_val),"wb") as f:
                   f.write((uploaded_video_val).getbuffer())
                   st.success("File Saved")
             #st.write(list_v_train)
             #st.write(path_title_list_train)
             st.write('fin de la boucle',i+1)    
                   
              #########sequences video###############     
                   
         for (i,z) in zip(list_v_train, path_title_list_train):
             fa.video_sequencer(i, z)
         for (i,z) in zip(list_v_val, path_title_list_val):
             fa.video_sequencer(i, z)

 

     
elif mode == 'Make the model':
     path_val_essaie = 'image_val_essaie/'
     path_train_essaie = 'image_train_essaie/'
     Nb_person = sum([len(d) for r, d, folder in os.walk(path_val_essaie)])
     
     st.write("You selected make the model")
     
     train_generator, validation_generator, x_test, y_test =  fa.train_test_split(path_train_essaie,  path_val_essaie)
     list_class = train_generator.class_indices
     with open("list_class", "wb") as fp:   #Pickling
         pickle.dump(list_class, fp)
     #####creation du modele#####
     layer_1 = st.sidebar.selectbox(
     'first layer conv',
     ('16','32', '64', '128'),key =1 )
     layer_1 = float(layer_1)
     #st.write('You selected:', layer_1)
     layer_2 = st.sidebar.selectbox(
     'second layer conv',
     ('32', '64', '128'),key =2 ) 
     layer_2 = float(layer_2)
     #st.write('You selected:', layer_2)
     layer_3 = st.sidebar.selectbox(
     'third layer conv',
     ('32', '64', '128','256','512'),key =3 )
     layer_3 = float(layer_3)
     #st.write('You selected:', layer_3)
     dropout_1 = st.sidebar.slider('choose first dropout', 0, 100, 10)
     
     if dropout_1 != 0:
          dropout_1 = (dropout_1 / 100)          
     #st.write('You selected:', dropout_1)
     dropout_2 = st.sidebar.slider('choose second dropout', 0, 100, 10)
     if dropout_2 != 0:
          dropout_2 = (dropout_2 / 100)
     #st.write('You selected:', dropout_2)
     kernel = st.sidebar.slider('choose kernel size', 1, 6, 1)
     #st.write('You selected:', kernel)
     stride = st.sidebar.slider('choose strides size', 1, 6, 1)
     #st.write('You selected:', stride)
     dense_1 = st.sidebar.selectbox(
     'first layer dense',
     ('1024', '2048', '4096'),key =4 ) 
     dense_1 = float(dense_1)
     #st.write('You selected:', dense_1)
     #st.write('You selected:', layers_1)
     dense_2 = st.sidebar.selectbox(
     'second layer dense',
     ('128', '256', '512'),key =5 )
     dense_2 = float(dense_2)
     #st.write('You selected:', dense_2)
     st.write('number of people',Nb_person)
     epochs = st.number_input('Number of epochs', step=1,min_value=1, max_value=100 )
     if st.button('calculate'):
         model = fa.make_model(layer_1,layer_2,layer_3,dropout_1,dropout_2,kernel,stride,dense_1,dense_2,Nb_person)
         
         #st.write(model)
         #st.write(sleep(randint(1,5))) # sleeping time 
         #epochs = st.number_input('Number of epochs', step=1,min_value=1, max_value=1000 )
         with st.spinner('Wait for it...'):
             history, model = fa.fit_model(model,train_generator,validation_generator,epochs)       
             st.success('Done!')
             ###save model#####      
         
         with st.spinner('saving the model'):
             model.save('my_model_1.h5')
             time.sleep(0.1)
             st.write('the model is saved')
             ###plot the model#####
         score = model.evaluate(x_test, y_test, verbose=0)
         st.write("Test loss:", score[0])
         st.write("Test accuracy:", score[1])
         fig1, fig2 = fa.plot_score_2(history)
         st.pyplot(fig1)
         st.pyplot(fig2)
     
else :
     st.write('You selected Mode test.')
     picture = st.camera_input("Take a picture")
     with open("list_class", "rb") as fp:   # Unpickling
         list_class = pickle.load(fp)
     list_class2 = []
     for i in list_class:
         list_class2.append(i)
     st.write('Voici la liste des classes:',list_class2)  
     st.write(list_class)
     if picture:
         
         model_1 = load_model('my_model_1.h5')
         name_folder = os.listdir('image_train_essaie/')
         pictures =  Image.open(picture)
         pictures.save('new_image.jpeg')
         img = Image.open('new_image.jpeg')
         # img.show()
         img1= img.convert("1").save("result.jpg")
         print(img1)
         st.image(img, caption='Super app')
         imgload = load_img('result.jpg', target_size=(150, 150))
         st.image(imgload, caption='Super app')
         imgarray = img_to_array(imgload)
         plt.imshow(imgarray / 255.)
         x = preprocess_input(np.expand_dims(imgarray.copy(), axis=0))
         st.write(x.shape)
         preds = model_1.predict(x)
         print(preds)
         st.write(name_folder[np.argmax(preds)])         
     uploaded_image = st.file_uploader("Choose a file")
     if uploaded_image:         
         model_1 = load_model('my_model_1.h5')
         #name_folder = os.listdir('image_train_essaie/')
         #st.write(name_folder)
         pictures =  Image.open(uploaded_image)
         pictures.save('new_image.jpeg')
         img = Image.open('new_image.jpeg')
         # img.show()         
         #img1= img.convert("1").save("result.jpg")
         #print(img1)
         st.image(img, caption='Super app')
         #imgload = load_img('result.jpg', target_size=(150, 150))
        # st.image(imgload, caption='Super app')
         #imgarray = img_to_array(imgload)
         #plt.imshow(imgarray / 255.)
         #x = preprocess_input(np.expand_dims(imgarray.copy(), axis=0))
         #st.write(x.shape)
         #preds = model_1.predict(x)
         img = tf.keras.preprocessing.image.img_to_array(img)
         img = tf.keras.preprocessing.image.smart_resize(img, (150, 150))
         img = tf.reshape(img, (-1, 150, 150, 3))
         prediction = model_1.predict(img/255)
         print(prediction)
         st.write(list_class2[np.argmax(prediction)]) 
     
# {path}\


# : 'image_train_essaie\\steve\\steve4.mp4'

    
#     img_path='new_image.jpeg' #dog
# # Define a new Model, Input= image 
# # Output= intermediate representations for all layers in the  
# # previous model after the first.
#     successive_outputs = [layer.output for layer in model.layers[1:]]
# #visualization_model = Model(img_input, successive_outputs)
#     visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# #Load the input image
#     img = imgload
# # Convert ht image to Array of dimension (150,150,3)
#     x   = img_to_array(img)                           
#     x   = x.reshape((1,) + x.shape)
# # Rescale by 1/255
#     x /= 255.0
# # Let's run input image through our vislauization network
# # to obtain all intermediate representations for the image.
#     successive_feature_maps = visualization_model.predict(x)
# # Retrieve are the names of the layers, so can have them as part of our plot
#     layer_names = [layer.name for layer in model.layers]
#     for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#         print(feature_map.shape)
#         if len(feature_map.shape) == 4:
    
#     # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
   
#             n_features = feature_map.shape[-1]  # number of features in the feature map
#             size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
#     # We will tile our images in this matrix
#         display_grid = np.zeros((size, size * n_features))
    
#     # Postprocess the feature to be visually palatable
#       #   for i in range(n_features):
#       #       x  = feature_map[0, :, :, i]
#       #       x -= x.mean()
#       #       x /= x.std ()
#       #       x *=  64
#       #       x += 128
#       #       x  = np.clip(x, 0, 255).astype('uint8')
#       # # Tile each filter into a horizontal grid
#       #       display_grid[:, i * size : (i + 1) * size] = x
# # Display the grid
#         scale = 20. / n_features
#         plt.figure( figsize=(scale * n_features, scale) )
#         plt.title ( layer_name )
#         plt.grid  ( False )
#         plt.imshow( display_grid, aspect='auto', cmap='viridis' )
#         st.pyplot()
    