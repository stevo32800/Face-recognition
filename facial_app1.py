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
from tensorflow.keras import regularizers



####################FONCTION POUR CREER DES IMAGES A PARTIR DE VIDEO#################
#IMPORTANT1: FONCTION UTILISER DANS LA BOUCLE COMMENTER
#IMPORTANT2: IL FAUT CREER 2 REPERTOIRE UN POUR LE TRAIN ET UN POUR LA VALIDATION
#IMPORTANT3: METTEZ LES IMAGES MANULELLEMENT DANS CHACUN DES DOSSIERS

def video_sequencer(namevid,namefold):
#videosequence

    import cv2
    vidcap = cv2.VideoCapture(f'{namevid}')
    success, image = vidcap.read()
    count = 1
    while success:       
        cv2.imwrite(f'{namefold}\image_{count}.jpg', image)    
        success, image = vidcap.read()
        print('Saved image ', count)
        count += 1
    print('process over!')
 
namevid = ['celia','silvain','steve'] # nom des vidéo que vous devez avoir dans le meme ficher que votre python  
namefold = ['celia','silvain','steve'] # nom des fichier ou vont se creer les images


###############IMPORTANT decommenter les 2 boucles pour creer des images de vos videos###############


# for (i,z) in zip(namevid, namefold):
#     print(i)
#     video_sequencer(i, z)

namevid_val = ['celia_val','silvain_val','steve_val']  
namefold_val = ['celia_val','silvain_val','steve_val']       

# for (i,z) in zip(namevid_val, namefold_val):
#     print(i)
#     video_sequencer(i, z)

############# FIN CREATION D'IMAGE #########################################



##############creation du df#################################

#etape optionnelle, vous pouvez la sauter

def create_df():
    filelist  = []

    for dirname, _, filenames in os.walk('image_facial_app'):
        for filename in filenames:
            filelist.append (os.path.join(dirname, filename))   

    print(len(filelist))

    labels_name = namefold
    Filepaths   = []
    labels = []
    image_datas = []

    for image_file in filelist:
        label = image_file.split(os.path.sep)[-2]
        #image_data = image_file.split(os.path.sep)[-1]
        if label in labels_name:
            image_datas = []
            Filepaths.append(image_file)
            labels.append(label)
            

    print(set(labels))

    len(Filepaths), len(labels)
    #creation du dataframe
    df = pd.DataFrame( list( zip (Filepaths, labels) ), columns = ['Filepath', 'Labels'] )
    return df
df = create_df()
#####################fin creation df#################################



##############CREATION DU SPLIT ET LA NORMALIZATION DES IMAGES#############
def train_test_split(path_train,path_test):
    batch_size = 10

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

    train_generator = train_datagen.flow_from_directory(
        path_train,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        
        )  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        path_test,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        )
    x_test, y_test = next(validation_generator)
    return train_generator, validation_generator, x_test, y_test


path_train = 'image_facial_app/'
path_test = 'image_val/'

#train_generator, validation_generator, x_test, y_test =  train_test_split(path_train, path_test)


#################################CREATION DU MODELE############################





def make_model(layer_1,layer_2, layer_3, dropout_1, dropout_2, kernel,stride,dense_1,dense_2,Nb_person):
    model = models.Sequential()
    model.add(layers.Conv2D(layer_1,(kernel,kernel),strides=(stride, stride), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(layer_2,(kernel, kernel),strides=(stride, stride), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(layer_3,(kernel, kernel),strides=(stride, stride), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
  
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_1,activation=('relu'))) 
    model.add(layers.Dropout(dropout_1))  
    model.add(layers.Dense(dense_2,activation=('relu'),  
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-5),
                       bias_regularizer = regularizers.l2(1e-3),
                       activity_regularizer=regularizers.l2(1e-3)))
    model.add(layers.Dropout(dropout_2))         
    model.add(layers.Dense(Nb_person,activation=('softmax')))

    model.summary()
    return model

layer_1 = 32
layer_2 = 64
layer_3 =128
dropout_1 = 0.5
dropout_2 = 0.3
kernel = 3
stride = 1
dense_1 = 2048
dense_2 = 512
Nb_person = 3

#model = make_model(layer_1,layer_2, layer_3, dropout_1, dropout_2, kernel,stride,dense_1,dense_2)

#print(model.summary())

##############################ENTRAINEMENT DU MODELE#################################

def fit_model(model,train_generator,validation_generator,epochs):
    epochs= epochs

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit_generator(train_generator,  
                              epochs= epochs, 
                              validation_data=validation_generator) 
                              
    return history, model

print('---------------first train--------------')
epochs = 1
#history, model = fit_model(model,train_generator,validation_generator,epochs)


##################SAUVEGARDE DU MODELE (TRES IMPORTANT POUR L'APP STREAMLIT )###############

#model.save('my_model.h5')


###################SCORE DU MODELE#################

#score = model.evaluate(x_test, y_test, verbose=0)
print('---------------accuracy of the model--------------')
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])

########################PLOTTING LES SCORES###########################


def plot_scrore(history):
# print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss =  history.history['loss']
    val_loss =  history.history['val_loss']
    epochs = range(1,len(acc)+1)
    
    fig1 = plt.plot(epochs,acc,'p',label ='entrainement')
    fig1 =plt.plot(epochs,val_acc,'g',label ='validation')
    fig1 =plt.xlabel("Epochs")
    fig1 =plt.ylabel("Loss")
    fig1 = plt.legend()
    fig1 = plt.figure()
    
    fig2 = plt.plot(epochs,loss,'b',label ='entrainement loss')
    fig2 =plt.plot(epochs,val_acc,'r',label ='validation loss')
    fig2 =plt.title('training and validation loss')
    fig2 = plt.legend()
    fig2 = plt.figure()
    return fig1, fig2

#plot_scrore(history)


def plot_score_2(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss'] 
  
    # plot results
    # accuracy
    fig1 = plt.figure(figsize=(10, 16))
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(f'\nTraining and Validation Accuracy. \nTrain Accuracy: {str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')
        # loss
    fig2 =plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)    
    plt.savefig('fig25_InceptionResNetV2.png')
    return fig1, fig2 

####################################TRAVAIL EN COURS###########################



#ETAGE DE L'ENCODADE L'IMAGE 
#IMPORTANT : CHANGER L'IMAGE DANS img_path

# img_path='steve_image10.jpeg' #dog
#kl#SS
# # Define a new Model, Input= image 
# # Output= intermediate representations for all layers in the  
# # previous model after the first.
# successive_outputs = [layer.output for layer in model.layers[1:]]
# #visualization_model = Model(img_input, successive_outputs)
# visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# #Load the input image
# img = load_img(img_path, target_size=(150, 150))
# # Convert ht image to Array of dimension (150,150,3)
# x   = img_to_array(img)                           
# x   = x.reshape((1,) + x.shape)
# # Rescale by 1/255
# x /= 255.0
# # Let's run input image through our vislauization network
# # to obtain all intermediate representations for the image.
# successive_feature_maps = visualization_model.predict(x)
# # Retrieve are the names of the layers, so can have them as part of our plot
# layer_names = [layer.name for layer in model.layers]
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#   print(feature_map.shape)
#   if len(feature_map.shape) == 4:
    
#     # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
   
#     n_features = feature_map.shape[-1]  # number of features in the feature map
#     size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
#     # We will tile our images in this matrix
#     display_grid = np.zeros((size, size * n_features))
    
#     # Postprocess the feature to be visually palatable
#     for i in range(n_features):
#       x  = feature_map[0, :, :, i]
#       x -= x.mean()
#       x /= x.std ()
#       x *=  64
#       x += 128
#       x  = np.clip(x, 0, 255).astype('uint8')
#       # Tile each filter into a horizontal grid
#       display_grid[:, i * size : (i + 1) * size] = x
# # Display the grid
#     scale = 20. / n_features
#     plt.figure( figsize=(scale * n_features, scale) )
#     plt.title ( layer_name )
#     plt.grid  ( False )
#     plt.imshow( display_grid, aspect='auto', cmap='viridis' )













