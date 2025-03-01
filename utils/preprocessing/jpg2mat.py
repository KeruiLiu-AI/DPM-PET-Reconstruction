import os
import numpy as np
from scipy import io
from skimage.io import imread
from skimage.transform import resize, rescale

# input and output directories
origin_dir = ''
dest_dir = ''
os.makedirs(dest_dir, exist_ok=True)

def plot_jpg_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".JPG"):
                path = os.path.join(root, file)

                # reading an image file
                img = imread(path)

                # make sure the image is two-dimensional (grayscale) and convert to grayscale if it is a color image
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)

                # creates an empty 3D array of size (1, 256, 128).
                img_combined = np.zeros((1, 256, 128))

                ground_truth = img

                # resize the image to 128x128 using the resize function
                ground_truth_resized = resize(ground_truth, (128, 128))

                # reduce the resolution of the image using the rescale function
                low_dose_pet = rescale(ground_truth_resized, scale=0.5, mode='reflect')
                low_dose_pet = resize(low_dose_pet, (128, 128))

                # the low-dose images and the ground truth images were merged
                low_dose_pet = low_dose_pet.transpose(0, 1)
                ground_truth_resized = ground_truth_resized.transpose(0, 1)
                img_combined[:, 0:128, :] = low_dose_pet
                img_combined[:, 128:256, :] = ground_truth_resized

                # save as a.mat file
                filename = file.replace(".JPG", "").replace('.', '_')
                print(f"Processing: {filename}")
                io.savemat(os.path.join(dest_dir, f'{filename}_0.mat'), {'img': img_combined})

# replace it with your directory path
directory_path = origin_dir
plot_jpg_files(directory_path)
