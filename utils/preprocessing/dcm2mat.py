import pydicom
import numpy as np
from scipy import io
import os
import pydicom
from skimage.transform import resize
from skimage.transform import rescale

origin_dir = './different_time_3_10'
dest_dir = './train_brain_pet_3_mat'
os.makedirs(dest_dir, exist_ok=True)


def plot_dcm_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                path = os.path.join(root, file)

                # read the DICOM file
                dcm = pydicom.dcmread(path, force=True)

                # extract the pixel array and plot the image
                img = dcm.pixel_array

                # creates an empty 3D array of size (1, 256, 128).
                img_combined = np.zeros((1, 256, 128))

                ground_truth = img
                # resize the image to 128x128 using the resize function
                ground_truth_resized = resize(ground_truth, (128, 128))

                # reduce the resolution of the image using the rescale function
                low_dose_pet = rescale(ground_truth_resized, scale=0.5, mode='reflect')
                low_dose_pet = resize(low_dose_pet, (128, 128))
                # the low-dose PET image and the ground truth image were merged
                low_dose_pet = low_dose_pet.transpose(0, 1)
                ground_truth_resized = ground_truth_resized.transpose(0, 1)
                img_combined[:, 0:128, :] = low_dose_pet
                img_combined[:, 128:256, :] = ground_truth_resized
                # save as a.mat file
                filename = file.replace(".dcm", "").replace('.', '_')
                print(filename)
                io.savemat(os.path.join(dest_dir, f'{filename}_0.mat'), {'img': img_combined})


# replace 'your_directory_path_here' with the path to your directory
directory_path = origin_dir
plot_dcm_files(directory_path)
