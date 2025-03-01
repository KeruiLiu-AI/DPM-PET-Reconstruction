import os
import shutil


def copy_and_rename_files(src_folder, dest_folder, target_count=128):
    # get all.mat files in the source folder
    mat_files = [f for f in os.listdir(src_folder) if f.endswith('.mat')]

    #make sure the desired folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # if the source folder doesn't have enough.mat files, copy the existing files first
    for i, mat_file in enumerate(mat_files):
        if i >= target_count:
            break
        src_path = os.path.join(src_folder, mat_file)
        dest_path = os.path.join(dest_folder, f"3_{i}.mat")
        shutil.copy(src_path, dest_path)

    # if the source folder does not have enough.mat files, keep copying files and renaming them until you reach the target number
    for i in range(len(mat_files), target_count):
        src_path = os.path.join(src_folder, mat_files[i % len(mat_files)])  # Recycle the files in the source folder
        dest_path = os.path.join(dest_folder, f"3_{i}.mat")
        shutil.copy(src_path, dest_path)

    print(f"Files have been copied and renamed to {target_count} .mat files in {dest_folder}.")


# example calls
src_folder = ''  # source folder path
dest_folder = ''  # destination folder path

copy_and_rename_files(src_folder, dest_folder, target_count=128)
