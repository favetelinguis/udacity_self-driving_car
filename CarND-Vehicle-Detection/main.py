from pyvision import lesson_functions

lesson_functions.draw_boxes()


def main():
    # TODO
    # read in blob of image paths
    # extract features from images
    # experiment with different features
    # normalize feature vector
    # train a classifier
    # persist good persifier with pickle
    # read in classifer
    # read in image and do one hog transform
    # perform sliding window over this hog image
    # classify image as car not car
    # clean up duplicates by centering in on only one image in overlapping windows
    # remove false posetive by constructing heatmap over multiple images
    # plot the bounding boxes in image
    # plot the road markings from last project in image
    # plog the heat maps as a small debug image
    print('main')


if __name__ == '__main__':
    main()
