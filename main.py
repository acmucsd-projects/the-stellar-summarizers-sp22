from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os


# created using https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class SumMeDataset(Dataset):
    def __init__(self, annotations_filename, img_dir, transform=None, target_transform=None):

        self.annotation_filename = annotations_filename
        self.annotation = pd.read_csv(annotations_filename, delimiter=' ')
        self.img_labels = self.annotation.iloc[:, 3]
        # print(self.img_labels)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'img_' + str(idx + 1).zfill(5) + '.jpg')
        print(img_path)
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


annotations_filename = './frames/annotation.txt'
img_root = './frames'
# num_lines = sum(1 for _ in open(annotations_filename))
# print(num_lines)

if __name__ == '__main__':

    videos_root = './frames/Bike_Polo'
    
    # dataset = VideoFrameDataset(
    #     root_path=videos_root,
    #     annotationfile_path='frames/annotation.txt',
    #     num_segments=1,
    #     frames_per_segment=num_lines,
    #     imagefile_template='img_{:0.5d}.jpg',
    #     transform=None,
    #     test_mode=False
    # )

    dataset = SumMeDataset(annotations_filename, videos_root)
    sample = dataset[1000]
    print(sample)
    plt.imshow(sample[0].permute(1, 2, 0))  # put channel data as last dimension
    plt.show()

    # print(sample.image)
    # plt.imshow(sample.image)
