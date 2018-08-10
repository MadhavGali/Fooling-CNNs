from keras.applications import VGG16
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from array import *
from PIL import Image
import keras.optimizers
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras import applications
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import image

BATCH_SIZE = 128
EPOCHS = 120
NUM_CLASSES = 2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3


LEARN_RATE = 1e-7
MOMENTUM = 0.6

#Filter layers
FILTER_SIZE_L1 = 7
NUM_FILTERS_L1 = 128

PADDING_SIZE = 1
PADDING_COLOR = (0, 0, 50)

def compile_Model(path):
    model = VGG16(weights = path, include_top=True, input_shape = (64,64,3), pooling = max, classes = 2)
    
    model.compile(loss='categorical_crossentropy', optimizer = 'SGD',metrics=['accuracy'])
    return model

def preprocess_data(path, path_PC, validationAR, validationPC):
    print("Preprocessing Started")
    #imlist = os.listdir(path)
    #imlist2 = os.listdir(path_PC)
    imlist_ValidAR = os.listdir(validationAR)
    imlist_ValidPC = os.listdir(validationPC)
    #create matrix to store all flattened images
    immatrix_ValidAR = np.array([np.array(Image.open(validationAR + im2)).flatten() for im2 in imlist_ValidAR], 'f')
    immatrix_ValidPC = np.array([np.array(Image.open(validationPC + im2)).flatten() for im2 in imlist_ValidPC], 'f')
    #immatrix = np.array([np.array(Image.open(path + im2)).flatten() for im2 in imlist], 'f')
    #immatrix1 = np.array([np.array(Image.open(validationpath + im2)).flatten() for im2 in imlist1], 'f')
    #immatrix2 = np.array([np.array(Image.open(path_PC + im2)).flatten() for im2 in imlist2], 'f')
    #num_samples is the total number of faces
    #num_samples = len(imlist)+len(imlist2)
    num_samples_validation = len(imlist_ValidAR) + len(imlist_ValidPC)
    #label is a 1D array which is of size num_samples and has values 0 in it
    #label = np.zeros((70000,), dtype=int)
    label_Validation = np.zeros((num_samples_validation,), dtype=int)
    #all the faces in the grayscale database is assigned the label 1, which means that 1 is aishwarya
    #label[0:30000] = 0
    #label[30001:60000] = 1
    label_Validation[0:183] = 0
    label_Validation[183:355] = 1
    #label1[5000:9998] = 1
    data_ValidationAR = shuffle(immatrix_ValidAR, random_state=2)
    data_ValidationPC = shuffle(immatrix_ValidPC, random_state=2)
    #data = shuffle(immatrix, random_state=2)
    #data2 = shuffle(immatrix2, random_state=2)
    #data1, Label1 = shuffle(immatrix1, label1, random_state=2)
    #print(label)
    #X_train= np.concatenate((data[0:25000], data2[0:25000]), axis=0)
    #Y_train = np.concatenate((label[0:25000], label[30000:55000]), axis=0)
    #X_test =  np.concatenate((data[25001:30000], data2[25001:30000]), axis=0)
    #Y_test = np.concatenate((label[25001:30000], label[55001:60000]), axis=0)
    #X_Valid , Y_Valid = data1[0:2], label1[0:13]
    X_Valid = np.concatenate((data_ValidationAR, data_ValidationPC), axis=0)
    Y_Valid = label_Validation
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    X_Valid = X_Valid.astype('float32')
   
    #normalizing the pixel values to be between 0 and 1, this helps the CNN converge faster
    #X_train /= 255
    #X_test /= 255
    X_Valid /= 255
    # convert labels to an indicator matrix
    #Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
    #Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)
    Y_Valid = np_utils.to_categorical(Y_Valid, NUM_CLASSES)
    #print("Shape of X_train = {0}".format(X_train[0].shape))
    #print("Shape of y_train = {0}".format(Y_train[0]))
    #print("Shape of Y_train = {0}".format(Y_train[0]))
    print("Preprocessing Started")
    return X_Valid, Y_Valid #X_train, X_test, Y_train, Y_test #X_Valid, Y_Valid
    
def train_model(model, X_train, X_test, Y_train, Y_test,fname):
    print("Training Started")

    X_train = np.expand_dims(X_train, axis=1)
    X_train = np.expand_dims(X_train, axis=1)
    X_train = np.reshape(X_train, (50000,64,64,3))
    print(X_train.shape) 
    X_test = np.expand_dims(X_test, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    print("xxtest")
    print(X_test.shape)
    X_test = np.reshape(X_test, (9998,64,64,3))
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    training = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test))
    model.save_weights(fname, overwrite=True)
    return model
    print("Training Finish")

def test_model(model, X_Valid, Y_Valid):
    X_Valid = np.expand_dims(X_Valid, axis=1)
    X_Valid = np.expand_dims(X_Valid, axis=1)
    X_Valid = np.reshape(X_Valid, (2,64,64,3))
    Y_Valid_results = model.predict(X_Valid)
    for i in range(len(Y_Valid)):
        if np.argmax(Y_Valid[i]) == np.argmax(Y_Valid_results[i]):
            true_count += 1
        else:
            false_count += 1
    print("True Percentage")
    print(true_count/len(Y_Valid))
    print("False Percentage")
    print(false_count/len(Y_Valid))


def main():
    data_path = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/trainingTheModel/AR1/'
    data_path_PC = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/trainingTheModel/PC/'
    validation_path_AR = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/trainingTheModel/ARtesting1/'
    validation_path_PC = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/trainingTheModel/PCtesting/'
    MODEL_WEIGHTS_PATH ='/home/stu11/s10/gg6549/CVIndependentStudy/version2/weights1_1.hdf5'

    FILTER_VISUALIZATION_PATH = "/home/stu11/s10/gg6549/CVIndependentStudy/version2/"    
    
    #Preprocess Data
    X_Valid, Y_Valid = preprocess_data(data_path, data_path_PC, validation_path_AR, validation_path_PC)
    model = compile_Model(MODEL_WEIGHTS_PATH)

    #Train model
    #model_trained =  train_model(model, X_train, X_test, Y_train, Y_test,MODEL_WEIGHTS_PATH)

    #Visualize filters layer 1
    #create_filter_visualization(model, FILTER_VISUALIZATION_PATH, 'filters_1', FILTER_SIZE_L1, NUM_FILTERS_L1)
   
    #Test model
    test_model(model, X_Valid, Y_Valid)
    

if __name__ == '__main__':
    main()
