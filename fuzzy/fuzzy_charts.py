import glob
import os
import shutil

# for file in glob.glob('data/charts/long/*'):
#     print(file)

# Recursively search for files in a directory 'data/charts'
for file in glob.glob('data/charts/short/**', recursive=True):
    # skip directories
    if not file.endswith('.png'):
        continue
    # figure out the name of parent directory
    parent_dir = file.split('/')[-2].lower()
    # downcase the name of parent directory
    print(parent_dir)
    print(file)
    # get basename of the file
    basename = os.path.basename(file)
    print(basename)
    # create new name for the file
    new_name = f"data/charts/{parent_dir}_{basename}"
    print(new_name)
    # Copy file to a new directory
    # if name matches 'accurac'
    if 'mean' in basename:
        shutil.copy(file, new_name)

# for file in glob.glob('data/charts/*'):
#     print(file)