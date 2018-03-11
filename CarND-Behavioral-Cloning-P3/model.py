#%%
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
#%%
def get_image(img_path, label, augment_flip):
    '''
    image_path path to image to be read
    label label belonging to picture
    augment_flip if we should flip the image and reverse the sign of the label
    output tuple with image list and label list
    '''
    images = []
    labels = []
    img = cv2.imread(img_path) # read images as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert images to RGB since images are read as RGB when predicting
    images.append(img)
    labels.append(label)
    if augment_flip: # Used if we want augmented data by flipping the images
        images.append(cv2.flip(img, 1))
        labels.append(label * -1.)
    return images, labels
def read_images(data, correction_factor=0, augment_flip=False):
    '''
    data = pandas dataframe with (Index, ImagePath, Steering)
    correction_factor = adjustment factor on the steering angel for left and right camera
    augment_flip = adds another image and measure where image is flipped and streating angle is multiplied with -1
    return X_train, y_train
    '''
    images = []
    labels = []
    for index, center_img_path, left_img_path, right_img_path, steering, throttle, break_val, speed in data.itertuples():
        img, label = get_image(center_img_path, steering, augment_flip)
        images += img
        labels += label
        if correction_factor is not 0:
            img, label = get_image(left_img_path, steering + correction_factor, augment_flip)
            images += img
            labels += label
            img, label = get_image(right_img_path, steering - correction_factor, augment_flip)
            images += img
            labels += label
    return np.array(images), np.array(labels)
def create_nvidia_model():
    '''
    model from https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    Preprocess image by normalizing to the range of 0-1 and mean center image data in a Lambda layer
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x /255. - .5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(58,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1)) # has no activation function
    return model
def run_model(model, data, labels, validation_split=.2, epoch=10, log_path = '/home/henke/repos/tblogs'):
    '''
    Compile the model and fits the data using the adam optimizer and mse
    model model to train
    data images in training set
    labels labels matching the training set
    validation_split how many % of data should be used for validation
    epoch for how many epoch should we run
    log_path where should the tensorflow log files go
    output the result of the training for each epoch the mse and mae
    '''
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    callbacks = [
        TensorBoard(log_dir=log_path)
    ]
    result = model.fit(data, labels, callbacks=callbacks, validation_split=validation_split, shuffle=True, epochs=epoch, batch_size=1024)
    return result
def save_model(model):
    '''
    Save the model to disk
    '''
    model.save('model.h5')
def save_images(X_train, y_train):
    '''
    Used to save images to disk
    '''
    for idx,img,l in zip(range(800), X_train, y_train):
        name = 'pic' + str(idx) + '_steering_' + str(l) + '.jpg'
        cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def main(path):
    '''
    Trains the model
    '''
    # Read the data
    df = pd.read_csv(path, names=['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed'])
    X_train, y_train = read_images(df, correction_factor=.2, augment_flip=False)
    model = create_nvidia_model()
    # Train the model
    run_model(model, X_train, y_train, epoch=5)
    save_model(model)
#%%
# Entrypoint for program
if __name__ == '__main__':
    main('/home/henke/repos/behavioraltrainingimages/driving_log.csv')