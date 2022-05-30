import os
import h5py
import numpy as np
from tqdm import tqdm


f = h5py.File('../frames.hdf5', 'w')
root = '../frames'

for folder in tqdm(os.listdir(root)):
    if folder == 'annotation.csv' or folder == '.DS_Store': 
        pass
    else:
        print(folder)
        group = f.create_group(folder) # create a subdir

        video_root = os.path.join(root, folder)
        print('Reading', video_root)

        for img_idx in range(len(os.listdir(video_root))):
            img_path = os.path.join(video_root, 'img_' + str(img_idx + 1).zfill(5) + '.jpg')
            print('Reading', img_path)
            with open(img_path, 'rb') as img: 
                np_binary = np.asarray(img.read())
                group.create_dataset(str(img_idx), data=np_binary)

f.close()

# test script
if __name__ == '__main__':
    f = h5py.File("frames.hdf5", "r")
    print(len(f['Jumps']))
    f.close()