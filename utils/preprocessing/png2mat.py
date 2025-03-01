import os
import matplotlib.pyplot as plt
import scipy.io


def png_to_mat(png_dir, mat_dir):
    # make sure the output directory exists
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    # iterate over all png files in the folder
    for filename in os.listdir(png_dir):
        if filename.endswith('.png'):
            png_path = os.path.join(png_dir, filename)
            # reading PNG images
            img = plt.imread(png_path)
            # construct the save path for the.mat file
            mat_filename = os.path.splitext(filename)[0] + '.mat'
            mat_path = os.path.join(mat_dir, mat_filename)
            # save the image data to a.mat file
            scipy.io.savemat(mat_path, {'img': img})
            print(f'Converted {filename} to {mat_filename}')


# set the path to the folder where the PNG image will be located and where the.mat file will be saved
png_directory = ''
mat_directory = ''

# call the function to perform the transformation
png_to_mat(png_directory, mat_directory)