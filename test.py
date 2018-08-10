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

def preprocess_data(path, path_PC, validationpath):
    print("Preprocessing Started")
    #imlist = os.listdir(path)
    #imlist2 = os.listdir(path_PC)
    imlist1 = os.listdir(validationpath)
    print("hello")
    # create matrix to store all flattened images
    #immatrix = np.array([np.array(Image.open(path + im2)).flatten() for im2 in imlist], 'f')
    immatrix1 = np.array([np.array(Image.open(validationpath + im2)).flatten() for im2 in imlist1], 'f')
    #immatrix2 = np.array([np.array(Image.open(path_PC + im2)).flatten() for im2 in imlist2], 'f')
    #num_samples is the total number of faces
    #num_samples = len(imlist)+len(imlist2)
    num_samples_validation = len(imlist1)
    #label is a 1D array which is of size num_samples and has values 0 in it
    #label = np.zeros((70000,), dtype=int)
    label1 = np.zeros((num_samples_validation,), dtype=int)
    #all the faces in the grayscale database is assigned the label 1, which means that 1 is aishwarya
    #label[0:30000] = 0
    #label[30001:60000] = 1
    label1[0:7223] = 0
    label1[7224:13000] = 1
    #data = shuffle(immatrix,random_state=2)
    #data2 = shuffle(immatrix2, random_state=2)
    data1, Label1 = shuffle(immatrix1, label1, random_state=2)
    #print(label)
    #X_train= np.concatenate((data[0:25000], data2[0:25000]), axis=0)
    #Y_train = np.concatenate((label[0:25000], label[30000:55000]), axis=0)
    #X_test =  np.concatenate((data[25001:30000], data2[25001:30000]), axis=0)
    #Y_test = (label[25001:34999])
    X_Valid , Y_Valid = data1[0:2], label1[0:13]
    #X_Valid = np.concatenate((data[10000:17223], data2[7224:13000]), axis=0)
    #Y_Valid = label1[0:13000]
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
    return X_Valid, Y_Valid#X_train, X_test, Y_train, Y_test, X_Valid, Y_Valid

def compile_model(weights_path):
    print("Compliling Model")
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(2, activation='softmax'))

    #model = applications.VGG16(weights=None, include_top=True, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    # Compile model    
    decay = LEARN_RATE/EPOCHS
    sgd = SGD(lr=LEARN_RATE, momentum=MOMENTUM, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
    model.load_weights(weights_path)
   
    #model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.RMSprop(lr = 1e-7),metrics=['accuracy'])
    print("Model Compliled")
    return model 

def train_model(model, X_train, X_test, Y_train, Y_test,fname):
    print("Training Started")

    X_train = np.expand_dims(X_train, axis=1)
    X_train = np.expand_dims(X_train, axis=1)
    X_train = np.reshape(X_train, (50000,64,64,3))
    
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
    print("Training Finish")
    print(training.history.keys())
       
    train_loss = training.history['loss']
    train_acc = training.history['acc']
    train_val_acc = training.history['val_acc']
    train_val_loss = training.history['val_loss']
    print("copied")
    
    print("training loss", train_loss)
    print("training accuaracy", train_acc)    
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i] = roc_curve(Y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    """
    plt.plot(train_acc, train_val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_45000_2.png')
    #plt.show()
    # summarize history for loss
    plt.plot(train_loss, train_val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('modelloss_45000_2.png')    
    #plt.show()
    return model
    
def test_model(model, X_Valid, Y_Valid):
    X_Valid = np.expand_dims(X_Valid, axis=1)
    X_Valid = np.expand_dims(X_Valid, axis=1)
    X_Valid = np.reshape(X_Valid, (2,64,64,3))
    Y_Valid_results = model.predict(X_Valid)
    print(Y_Valid_results)
    """
    true_count = 0
    false_count = 0
    print(Y_Valid.shape)
    print(Y_Valid)
    print(Y_Valid_results.shape)
    print(Y_Valid_results)
    for i in range(len(Y_Valid)):
        if np.argmax(Y_Valid[i]) == np.argmax(Y_Valid_results[i]):
            true_count += 1
        else:
            false_count += 1
    print("True Percentage")
    print(true_count/len(Y_Valid))
    print("False Percentage")
    print(false_count/len(Y_Valid))
    """


def test_single_image(image):
    return 0
    
def normalize(img_array):        
    for i in range(0,3):
        max_val = 0
        min_val = 100000
        for j in range(0,7):
            for k in range(0,7):
                if(img_array[j,k,i] > max_val):
                    max_val = img_array[j,k,i]
                if(img_array[j,k,i] < min_val):
                    min_val = img_array[j,k,i]
        img_array[:,:,i] = img_array[:,:,i] - min_val
        img_array[:,:,i] = img_array[:,:,i] / max_val
    return img_array

def visualize_single_filter(img_array):
    img = Image.new("RGB", (7, 7),'white')
    pix = img.load()
  
    for i in range(0, 7):
        for j in range(0, 7):
            red = int(img_array[i][j][0])
            green = int(img_array[i][j][1])
            blue = int(img_array[i][j][2])
            pix[i,j] = (red, green, blue)
  
    return img

#Requires a num_filter divisible by 16
def create_filter_visualization(model, visualized_filter_path, visualized_filename, filter_size, num_filter):
    row = 0
    column = 0
    
    img_col = 16
    img_row = int(num_filter/16)
    
    final_image = Image.new('RGB', ((filter_size*img_col)+(PADDING_SIZE*img_col-2), 
                                    (filter_size*img_row)+(PADDING_SIZE*img_row-2)),
                                    PADDING_COLOR)
    print("WIDTH: " + str((filter_size*img_col)+(PADDING_SIZE*img_col-1)))
    print("HEIGHT: " + str((filter_size*img_row)+(PADDING_SIZE*img_row-1)))
    
    for i in range(num_filter):
        img_array = model.layers[0].get_weights()[0]
        v = normalize(img_array[:,:,:,i])
        v = v * 255
        single_filter = visualize_single_filter(v)
        final_image.paste(single_filter, (column, row))
        column += filter_size + PADDING_SIZE
        if(column > ((img_col-1) * (filter_size + PADDING_SIZE))):
            column = 0
            row = row + filter_size + PADDING_SIZE
            
    final_image.save(visualized_filter_path + visualized_filename + '.png')
    print("Filter Visualization Saved")            

#Run an image through the network and see which route it takes and 
#Visualize this as a heatmap
def heatmap(image):
    result = test_single_image()
    return result

def main():
    data_path = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/AR1/'
    data_path_PC = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/PC/'
    validation_path = '/home/stu11/s10/gg6549/CVIndependentStudy/version2/ARShuffle/Adversary/'
    MODEL_WEIGHTS_PATH ='/home/stu11/s10/gg6549/CVIndependentStudy/version2/weights1_1.hdf5'

    FILTER_VISUALIZATION_PATH = "/home/stu11/s10/gg6549/CVIndependentStudy/version2/"    
    
    #Preprocess Data
    #X_train, X_test, Y_train, Y_test, 
    #X_Valid, Y_Valid = preprocess_data(data_path, data_path_PC, validation_path)
    model = compile_model(MODEL_WEIGHTS_PATH)

    #Train model
    #model_trained =  train_model(model, X_train, X_test, Y_train, Y_test,MODEL_WEIGHTS_PATH)

    #Visualize filters layer 1
    #create_filter_visualization(model, FILTER_VISUALIZATION_PATH, 'filters_1', FILTER_SIZE_L1, NUM_FILTERS_L1)
   
    #Test model
    test_model(model, X_Valid, Y_Valid)
    

if __name__ == '__main__':
    main()
