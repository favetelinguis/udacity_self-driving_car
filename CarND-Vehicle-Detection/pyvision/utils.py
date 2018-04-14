import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

# -- IO util functions ---
def get_images_paths(path, pattern):
    '''
    Get path to all images in a specific folder.
    '''
    imgs = glob.glob(path + '/' + pattern)
    return imgs

def get_image(path):
    '''
    Load an image and return it in rgb as default
    cv2 reads images in BGR so we also need to convert the image
    '''
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def output_image(img, out_path, name):
    '''
    Read example images and print them to example map
    '''
    cv2.imwrite(out_path + '/' + name + '.png',img)

# --- Image util functions ---
def convert_to_grayscale(img):
    '''
    Convert image to grayscale
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def convert_to_hls(img):
    '''
    convert image to hls
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def convert_to_hsv(img):
    '''
    convert image to hsv
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
def convert_to_lab(img):
    '''
    convert image to lab
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def plot_list(nested_imgs, nested_titles, title='', size=(10,10)):
    '''
    convinenece function to show a nested list of images
    '''
    rows = len(nested_imgs)
    cols = len(nested_imgs[0])
    cmap = None
    fig, axes = plt.subplots(rows, cols, figsize=size)

    for idy, imgs in enumerate(nested_imgs):
        for idx, img in enumerate(imgs):
            ax = None
            if rows > 1 and cols > 1:
                ax = axes[idy, idx]
            if rows is 1:
                ax = axes[idx]
            if cols is 1:
                ax = axes[idy]
            img_title = nested_titles[idy][idx]
            if len(img.shape) < 3 or img.shape[-1] < 3:
                cmap = "gray"
                img = np.reshape(img, (img.shape[0], img.shape[1]))
            ax.imshow(img, cmap=cmap)
            ax.set_title(img_title)
            ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1)
    fig.tight_layout()
    plt.show()


def apply_and_plot(fun, img_paths, size=(20, 20)):
    imgs = []
    titles = []
    for img_path in img_paths:
        img = get_image(img_path)
        imgs.append([fun(img)])
        titles.append([img_path.rsplit('/', 1)[-1]])
    plot_list(imgs, titles, size=size)
