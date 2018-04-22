import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import lesson_functions


def get_data():
    vehicle_imgs = glob.glob('/Users/henriklarsson/Downloads/vehicles/**/*.png')
    nonvehicle_imgs = glob.glob('/Users/henriklarsson/Downloads/non-vehicles/**/*.png')
    #cars = []
    #notcars = []
    # Make sure we have the same number vehicles and non vehicles samples
    #for vehicle, nonvehicle in zip(vehicle_imgs, nonvehicle_imgs):
    #    notcars.append(nonvehicle)
    #    cars.append(vehicle)
    return vehicle_imgs, nonvehicle_imgs


def training(color_space='HLS', spatial_size=(16, 16), hist_bins=32,
                           orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                           spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    color_space Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_channel Can be 0, 1, 2, or "ALL"
    '''
    cars, notcars = get_data()
    print('Num cars: {} Num non-cars: {}'.format(len(cars), len(notcars)))

    t = time.time()
    car_features = lesson_functions.extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                     spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = lesson_functions.extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                     spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    return svc, X_scaler


def train_with_grid_search(color_space='YUV', spatial_size=(32, 32), hist_bins=32,
                           orient=11, pix_per_cell=16, cell_per_block=2, hog_channel='ALL',
                           spatial_feat=False, hist_feat=False, hog_feat=True):
    parameters = {'C': [1, 100, 1000, 10000]}
    svc = LinearSVC()
    clf = GridSearchCV(svc, parameters, n_jobs=3, verbose=5)

    cars, notcars = get_data()

    print('Extracting features')
    car_features = lesson_functions.extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                     spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = lesson_functions.extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                     spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print('Starting to fit model')
    clf.fit(X, y)
    print('Best parameters found: ' + str(clf.best_params_))


if __name__ == '__main__':
    opt = sys.argv[1]
    if 'grid' in opt:
        print('Searching best parameters')
        params = train_with_grid_search()
        print(params)
    else:
        print('Starting training')
        clf, scaler = training()
        from sklearn.externals import joblib
        print('Persisting model')
        joblib.dump(clf, '../model.pkl')
        joblib.dump(scaler, '../scaler.pkl')
