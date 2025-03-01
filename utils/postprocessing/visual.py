import os
from medpy.io import load, save
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim


import matplotlib.pyplot as plt
import numpy as np

# Assuming `SPETimg` and `EPETimg` are 3D numpy arrays with shape (128, 128, 128)

# Function to normalize the intensity values for visualization
def normalize_slice(slice):
    return (slice - np.min(slice)) / (np.max(slice) - np.min(slice))

# Select middle slices from each dimension
def get_middle_slices(chann, img):
    slices = []
    for axis in range(3):
        # Use np.take to select the slice in the middle along each axis
        slice = np.take(img, chann, axis=axis % 3)
        slices.append(normalize_slice(slice))
    return slices

if __name__ == '__main__':
    
    # path is your result root
    path = r'/experiments/sr_ffhq_241010_212629/results'
    hr_path = []
    result_path = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in files:
            if file.endswith('hr.img'):
                hr_path.append(os.path.join(root, file))
            if file.endswith('result.img'):
                result_path.append(os.path.join(root, file))

    total_psnr = []
    total_ssim = []
    total_nmse = []
    for i in range(len(hr_path)):
        
        SPETimg,_ = load(hr_path[i])
        EPETimg,_ = load(result_path[i])
        
        chann, weight, height = EPETimg.shape
        for begin_chann in np.arange(0, chann):
            # Get middle slices for each image
            SPET_slices = get_middle_slices(begin_chann, SPETimg)
            EPET_slices = get_middle_slices(begin_chann, EPETimg)

            # Plotting
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for j, (SPET_slice, EPET_slice) in enumerate(zip(SPET_slices, EPET_slices)):
                # SPET Image slice
                axes[j].imshow(SPET_slice, cmap='gray')
                axes[j].set_title(f'SPET {["Axial", "Coronal", "Sagittal"][j]} Slice')
                axes[j].axis('off')
                # EPET Image slice
                axes[j+3].imshow(EPET_slice, cmap='gray')
                axes[j+3].set_title(f'EPET {["Axial", "Coronal", "Sagittal"][j]} Slice')
                axes[j+3].axis('off')

            plt.tight_layout()
            img_path = os.path.join(path, f'{i}')
            os.makedirs(img_path, exist_ok=True)
            true_img_path = os.path.join(img_path, f'image_{begin_chann}.png')
            plt.savefig(true_img_path)
            plt.close('all')
            del fig
            
            print(f'{true_img_path} has been written into.')
        
        chann, weight, height = EPETimg.shape
        for c in range(chann): 
            for w in range(weight):  
                for h in range(height):
                    if EPETimg[c][w][h] <= 0.05:
                        EPETimg[c][w][h] = 0
                        SPETimg[c][w][h] = 0
        y = np.nonzero(EPETimg)
        im1_1 = SPETimg[y]
        im2_1 = EPETimg[y]

        dr = np.max([im1_1.max(), im2_1.max()]) - np.min([im1_1.min(), im2_1.min()])
        cur_psnr = psnr(im1_1, im2_1, data_range=dr)
        cur_ssim = ssim(SPETimg, EPETimg, multi_channel=1, data_range=dr)
        cur_nmse = nmse(im1_1, im2_1) ** 2
        print('PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(cur_psnr, cur_ssim, cur_nmse))

        total_psnr.append(cur_psnr)
        total_ssim.append(cur_ssim)
        total_nmse.append(cur_nmse)
    avg_psnr = np.mean(total_psnr)
    avg_ssim = np.mean(total_ssim)
    avg_nmse = np.mean(total_nmse)
    print('Avg. PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr, avg_ssim, avg_nmse))
