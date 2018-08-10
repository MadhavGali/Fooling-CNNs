import numpy as np
from keras import backend as K     
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense


#Load the dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
       
X_train_1 = X_train[0:20000]
X_train_2 = X_train[20000:40000]
X_matrix_train = X_train[40000:50000]

Y_train_1 = Y_train[0:20000]
Y_train_2 = Y_train[20000:40000]
Y_matrix_train = Y_train[40000:50000]

#Create the model
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


DATA_FILE_PATH = '/home/tsz2759/Documents/Research/Extracted Features/'

def get_extracted_features(model, index , image):
    get_extracted = K.function([model.layers[0].input, K.learning_phase()], [model.layers[index].output,])
    features = get_extracted([image,0])
    return features

X_train_1_extracted = np.empty(shape=(20000, 10))
X_train_2_extracted = np.empty(shape=(20000, 10))
X_matrix_extracted = np.empty(shape=(10000, 10))
X_test_extracted = np.empty(shape=(10000, 10))

#Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

#Training model for train_1
model.fit(X_train_1 / 255.0, to_categorical(Y_train_1),
          batch_size=128,
          shuffle=True,
          epochs=EPOCHS) 

#X_train_1 
for i in range(int(len(X_train_1)/BATCH_SIZE)):
    features = (get_extracted_features(model, EXTRACTED_INDEX, X_train_1))
    features = np.array(features)

np.savetxt(DATA_FILE_PATH + 'extracted_features_train_1.txt', X_train_1_extracted)
test_data_1_load = np.loadtxt(DATA_FILE_PATH + 'extracted_features_train_1.txt')

if X_train_1_extracted.all() == test_data_1_load.all():
    print('Extracted features for Train_1 were saved and loaded correctly.')
    

             
#X_train_2
for i in range(int(len(X_train_2)/BATCH_SIZE)):
    features = (get_extracted_features(model, EXTRACTED_INDEX, X_train_2[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]))
    features = np.array(features)
    for j in range(BATCH_SIZE):
        X_train_2_extracted[i*BATCH_SIZE+j,:] = features[:,j,:]

np.savetxt(DATA_FILE_PATH + 'extracted_features_train_2.txt', X_train_2_extracted)
test_data_2_load = np.loadtxt(DATA_FILE_PATH + 'extracted_features_train_2.txt')

if X_train_2_extracted.all() == test_data_2_load.all():
    print('Extracted features for Train_2 were saved and loaded correctly.')
    
    
#Training model for confusion matricies
model.fit(X_matrix_train / 255.0, to_categorical(Y_matrix_train),
          batch_size=128,
          shuffle=True,
          epochs=EPOCHS) 
    
#X_confusion_matricies
for i in range(int(len(X_matrix_train)/BATCH_SIZE)):
    features = (get_extracted_features(model, EXTRACTED_INDEX, X_matrix_train[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]))
    features = np.array(features)
    for j in range(BATCH_SIZE):
        X_matrix_extracted[i*BATCH_SIZE+j,:] = features[:,j,:]

np.savetxt(DATA_FILE_PATH + 'extracted_features_matrix.txt', X_matrix_extracted)
matrix_data_load = np.loadtxt(DATA_FILE_PATH + 'extracted_features_matrix.txt')

if X_matrix_train.all() == matrix_data_load.all():
    print('Extracted features for Confusion Matrix Validation were saved and loaded correctly.')
    
#Training model for test
model.fit(X_test / 255.0, to_categorical(Y_test),
          batch_size=128,
          shuffle=True,
          epochs=EPOCHS)

#X Test
for i in range(int(len(X_test)/BATCH_SIZE)):
    features = (get_extracted_features(model, EXTRACTED_INDEX, X_test[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]))
    features = np.array(features)
    for j in range(BATCH_SIZE):
        X_test_extracted[i*BATCH_SIZE+j,:] = features[:,j,:]
        
np.savetxt(DATA_FILE_PATH + 'extracted_features_test.txt', X_test_extracted)
matrix_data_load = np.loadtxt(DATA_FILE_PATH + 'extracted_features_test.txt')

if X_test_extracted.all() == matrix_data_load.all():
    print('Extracted features for Test Data were saved and loaded correctly.')
    
    
def main():
        
    
    