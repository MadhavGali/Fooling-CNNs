import numpy as np
from PIL import Image
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

IMAGE_WIDTH, IMAGE_HEIGHT, IMG_CHANNELS = 64, 64, 3

EPSILON = 0.025

#Model Parameters 
EPOCHS = 100
LEARN_RATE = 0.0001
MOMENTUM = 0.3
NUM_CLASSES = 2

#Fooling Network
REQUIRED_COST = 0.9

#Image Generation
CATEGORIES = 1
IMG_INSTANCES = 3

PRIYANKA_CHOPRA = 1
AISHWARYA_RAI = 0

def compile_model(weights_path):
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

    # Compile model    
    decay = LEARN_RATE/EPOCHS
    sgd = SGD(lr=LEARN_RATE, momentum=MOMENTUM, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model.load_weights(weights_path)
    
    return model 

def get_above_threshold(image):
    return image + EPSILON

def get_below_threshold(image):
    return image - EPSILON

def create_fool_image(model, fake_image_class, true_image):

    print("reached")
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output
        
    #Adjust to fake image class
    cost_function = model_output_layer[0, fake_image_class] 

    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    get_costs_and_gradients = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0
    
    original_image= image.img_to_array(true_image)
    
    original_image /= 255
    original_image-= 0.5
    original_image*= 2.0        

    original_image = np.expand_dims(original_image, axis=0)
    
    modified_image = np.copy(original_image)

    #previous_cost = 100
    
    #CONVERGENCE_RATE = 0.1

    while cost < REQUIRED_COST:
        cost, gradients = get_costs_and_gradients([modified_image, 0])    
        modified_image += (gradients * LEARN_RATE)        

        #current_cost = cost 
        
        #if current_cost - previous_cost < CONVERGENCE_RATE:
        #    LEARN_RATE = LEARN_RATE / 1.005
        #    CONVERGENCE_RATE = CONVERGENCE_RATE / 2
            
        #f LEARN_RATE < 0.01:
        #   LEARN_RATE = 0.01
            
        #previous_cost = current_cost            
        
        #Clip the function to bound its changes
        modified_image = np.clip(modified_image, get_below_threshold(modified_image), get_above_threshold(modified_image))
        modified_image = np.clip(modified_image, -1.0, 1.0)
        print(str(cost*100) + ' | LR: ' + str(LEARN_RATE)) #TODO remove this later
        
    modified_image = modified_image[0]
    modified_image /= 2.0
    modified_image += 0.5
    modified_image *= 255
    
    result = Image.fromarray(modified_image.astype(np.uint8))
    return result

def classify_image(test_image, model):    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    test_image = test_image / 255
    test_image = test_image - 0.5
    test_image = test_image * 2
    
    #print("Making prediction..")
    predictions = model.predict(test_image)
    #print("Made prediction")
    print(predictions)

#Shows the noise pictorially that is added to the image to make
#it misclassify as the targeted fake class
def show_noise(original, adversarial, name):
    result = Image.new('RGB', size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    result_pix = result.load()
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            org_red, org_green, org_blue = original.getpixel((i,j))
            adv_red, adv_green, adv_blue = adversarial.getpixel((i,j))
            red_result = abs(org_red - adv_red)
            green_result = abs(org_green - adv_green)
            blue_result = abs(org_blue - adv_blue)
            result_pix[i, j] = (red_result, green_result, blue_result, 255)
    result = normalize(result)
    result.save(name)
    return result
    
def normalize(image):
    min_red = 255
    min_green = 255
    min_blue = 255
    max_red = 0
    max_green = 0
    max_blue = 0
    
    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            red, green, blue = image.getpixel((i,j))
            if red > max_red:
                max_red = red
            if red < min_red:
                min_red = red
            if green > max_green:
                max_green = green
            if green < min_green:
                min_green = green
            if blue > max_blue:
                max_blue =blue
            if blue < min_blue:
                min_blue = blue
                
    image_pix = image.load()
    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            red, green, blue = image.getpixel((i,j))
            modified_red = (red - min_red) / max_red * 255
            modified_green = (green - min_green) / max_green * 255
            modified_blue = (blue - min_blue) / max_blue * 255
            image_pix[i, j] = (int(modified_red), int(modified_green), int(modified_blue), 255)
    return image
    
#Shows the probabilies for each 
def show_probability(images, model):
    img_names = ['peppers', 'castle', 'violin', 'killer whale']
    inst_names = ['original', 'adversarial noise', 'combined']
    for i in range(CATEGORIES):
        for j in range(IMG_INSTANCES):
            print(inst_names[j] + ' ' + img_names[i] + ':')
            classify_image(images[i][j], model)
            print('\n')
            
#Make a function that has the original image, noise for that image, 
#and resulting adverary image for all four images 
def show_results(images, fn):
    space_amount = 3
    result = Image.new('RGB', size=(IMAGE_WIDTH * IMG_INSTANCES + (IMG_INSTANCES-1*space_amount),                                 
                                    IMAGE_HEIGHT * CATEGORIES + (CATEGORIES-1*space_amount)))
    for i in range(CATEGORIES):
        for j in range(IMG_INSTANCES):
            img = images[i][j]
            result.paste(img, (j*IMAGE_WIDTH + (j*space_amount), i*IMAGE_HEIGHT + (i*space_amount)))
    result.show()
    result.save(fn)


def main():
    WEIGHTS_PATH = 'weights_orig_model.hdf5'
    
    IMAGE_PATH_AR = 'PCafter/'
    IMAGE_PATH_PC = 'PCafter/'
    IMAGE_PATH_AR_Adversary = 'PCShuffle/Adversary/'
    IMAGE_PATH_PC_Adversary = 'PCShuffle/Adversary/'

    model = compile_model(WEIGHTS_PATH)
    print(model.summary())
    """
    for i in range(40, 184):
        print(i)
        classify_image(image.load_img(IMAGE_PATH_AR + 'image'+str(100)+'.jpg', target_size=(64,64)), model)
        classify_image(image.load_img(IMAGE_PATH_AR + 'Adversary_'+str(100)+'.png', target_size=(64,64)), model)
       
        #print("Before")
        classify_image(image.load_img(IMAGE_PATH_AR + 'image'+str(i+1)+'.jpg', target_size=(64,64)), model)
        
        img_AR = image.load_img(IMAGE_PATH_AR + 'image'+str(i+1) + '.jpg', target_size=(64,64))    
        fooled_image_AR = create_fool_image(model, PRIYANKA_CHOPRA, img_AR)
        #print("Before")
        #classify_image(img_AR, model)
        fooled_image_AR.save(IMAGE_PATH_AR + 'Adversary_' + str(i+1) + '.png')
        fooled_noise_AR = show_noise(img_AR, fooled_image_AR, IMAGE_PATH_AR + 'Noise_' + str(i+1) + '.png')
        
        final_results_AR = []
        final_results_AR.append([img_AR, fooled_noise_AR, fooled_image_AR])
        show_results(final_results_AR, IMAGE_PATH_AR + 'Combined_' + str(i+1) + '.png')
        
        print("after")
        #classify_image(image.load_img(IMAGE_PATH_AR_Adversary + 'Adversary_'+str(i+1)+'.png', target_size=(64,64)), model)
        
        classify_image(fooled_image_AR, model)
        
        #print("PC")
        #print("Before")
        #classify_image(image.load_img(IMAGE_PATH_PC + 'image'+str(i+1)+'.jpg', target_size=(64,64)), model)
        #print("after")
        #classify_image(image.load_img(IMAGE_PATH_PC_Adversary + 'Adversary_'+str(i+1)+'.png', target_size=(64,64)), model)
       
        img_PC = image.load_img(IMAGE_PATH_PC + 'image'+str(i+1) + '.jpg', target_size=(64,64))    
        fooled_image_PC = create_fool_image(model, AISHWARYA_RAI, img_PC)    
        print("before")
        classify_image(img_PC, model)
        fooled_image_PC.save(IMAGE_PATH_PC + 'Adversary_' + str(i+1) + '.png')        
        fooled_noise_PC = show_noise(img_PC, fooled_image_PC, IMAGE_PATH_PC + 'Noise_' + str(i+1) + '.png')
        print("After")
        classify_image(fooled_image_PC, model)
        final_results_PC = []
        final_results_PC.append([img_PC, fooled_noise_PC, fooled_image_PC])
        show_results(final_results_PC, IMAGE_PATH_PC + 'Combined_' + str(i+1) + '.png')
        """        
if __name__ == '__main__':
    main()
